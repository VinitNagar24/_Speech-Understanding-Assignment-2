"""
part3_tts/task3_1_voice_embedding.py
--------------------------------------
Extracts a high-dimensional speaker embedding (x-vector / d-vector)
from a 60-second reference recording of your own voice.

Two backends are supported:
  1. SpeechBrain ECAPA-TDNN (x-vector) — preferred
  2. Fallback: simple statistics-based d-vector (no extra dependencies)

Usage:
    python task3_1_voice_embedding.py \
        --audio data/student_voice_ref.wav \
        --output output/speaker_embedding.npy
"""

import os
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Tuple

TARGET_SR = 16000   # SpeechBrain ECAPA-TDNN expects 16kHz
EMBED_DIM  = 192    # ECAPA-TDNN output dimension


# ── Fallback d-vector (statistics-based, no SpeechBrain needed) ──────────────

class StatsDVector:
    """
    Simple d-vector: concatenation of [mean, std, min, max] of MFCC features.
    Dimension = 4 * n_mfcc (default 160-D for n_mfcc=40).
    Serves as a fallback when SpeechBrain is not available.
    """

    def __init__(self, n_mfcc: int = 40, sr: int = TARGET_SR):
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 80},
        )

    def extract(self, waveform: torch.Tensor) -> np.ndarray:
        """
        waveform : (1, T) float32 tensor
        returns  : (4*n_mfcc,) numpy float32 array
        """
        with torch.no_grad():
            mfcc = self.mfcc_transform(waveform)  # (1, n_mfcc, time)
            mfcc = mfcc.squeeze(0)                  # (n_mfcc, time)

        mean = mfcc.mean(dim=1).numpy()
        std  = mfcc.std(dim=1).numpy()
        mn   = mfcc.min(dim=1).values.numpy()
        mx   = mfcc.max(dim=1).values.numpy()

        dvec = np.concatenate([mean, std, mn, mx])
        # L2 normalize
        dvec = dvec / (np.linalg.norm(dvec) + 1e-8)
        return dvec.astype(np.float32)


# ── SpeechBrain x-vector extractor ───────────────────────────────────────────

class XVectorExtractor:
    """
    Wraps SpeechBrain's ECAPA-TDNN pretrained model for x-vector extraction.
    Falls back to StatsDVector if SpeechBrain is not installed.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_speechbrain()

    def _load_speechbrain(self):
        try:
            from speechbrain.pretrained import EncoderClassifier
            print("[XVector] Loading SpeechBrain ECAPA-TDNN ...")
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device},
            )
            print("[XVector] SpeechBrain model loaded ✓")
        except ImportError:
            print("[XVector] SpeechBrain not found — using fallback d-vector")
            self.model = None
        except Exception as e:
            print(f"[XVector] SpeechBrain failed ({e}) — using fallback d-vector")
            self.model = None

    def extract(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        """
        waveform : (1, T) or (T,) tensor
        sr       : sample rate of waveform
        returns  : 1-D numpy array (192-D for ECAPA, 160-D for fallback)
        """
        # Ensure mono (1, T)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        # Resample to 16kHz
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            waveform = resampler(waveform)

        if self.model is not None:
            with torch.no_grad():
                wav = waveform.squeeze(0).unsqueeze(0).to(self.device)
                embedding = self.model.encode_batch(wav)  # (1, 1, 192)
                embedding = embedding.squeeze().cpu().numpy()
            # L2 normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding.astype(np.float32)
        else:
            # Fallback
            dv = StatsDVector(sr=TARGET_SR)
            return dv.extract(waveform)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_speaker_embedding(
    audio_path: str,
    output_path: str = "output/speaker_embedding.npy",
    required_duration: float = 60.0,
) -> np.ndarray:
    """
    Load 60s reference audio and extract a speaker embedding.

    Parameters
    ----------
    audio_path        : path to student_voice_ref.wav (≥60s)
    output_path       : where to save the .npy embedding
    required_duration : minimum expected duration in seconds

    Returns
    -------
    embedding : 1-D float32 numpy array
    """
    waveform, sr = torchaudio.load(audio_path)
    duration = waveform.shape[-1] / sr
    print(f"[XVector] Loaded {audio_path}  ({duration:.1f}s, {sr}Hz)")

    if duration < required_duration - 1.0:
        print(f"[XVector] WARNING: audio is {duration:.1f}s, expected ≥{required_duration}s")

    extractor = XVectorExtractor()
    embedding = extractor.extract(waveform, sr)

    print(f"[XVector] Embedding shape: {embedding.shape}  norm: {np.linalg.norm(embedding):.4f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, embedding)
    print(f"[XVector] Saved embedding → {output_path}")
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--audio",  required=True, help="Path to student_voice_ref.wav")
    p.add_argument("--output", default="output/speaker_embedding.npy")
    args = p.parse_args()

    emb = extract_speaker_embedding(args.audio, args.output)
    print(f"[XVector] Embedding (first 10 dims): {emb[:10]}")
