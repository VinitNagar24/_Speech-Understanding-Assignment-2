"""
part3_tts/task3_2_prosody_warping.py
--------------------------------------
Extracts F0 (fundamental frequency) and Energy contours from the
professor's lecture, then applies Dynamic Time Warping (DTW) to
map these prosodic features onto synthesized speech.

Pipeline:
  1. Extract F0 + Energy from reference (professor's audio).
  2. Extract F0 + Energy from synthesized (student's TTS output).
  3. DTW align both contours.
  4. Warp synthesized audio's pitch track to match reference.
  5. Reconstruct waveform using PSOLA-like pitch shifting.

Mathematical formulation:
  DTW(X, Y) = min_{π} Σ d(x_{π(t)}, y_t)
  where π is the warping path minimizing cumulative distance.
"""

import os
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import scipy.signal
import scipy.interpolate
from typing import Optional, Tuple


SR          = 22050
HOP_LENGTH  = 256     # frame hop for F0 extraction
FRAME_LEN   = 1024    # frame length
F0_MIN      = 60.0    # Hz — minimum F0 (male voice lower bound)
F0_MAX      = 500.0   # Hz — maximum F0 (female upper bound)


# ── F0 Extraction (YIN algorithm) ────────────────────────────────────────────

def extract_f0_yin(waveform: np.ndarray, sr: int,
                   hop: int = HOP_LENGTH, fmin: float = F0_MIN, fmax: float = F0_MAX,
                   threshold: float = 0.1) -> np.ndarray:
    """
    YIN fundamental frequency estimator.
    Returns F0 array (Hz) per frame; 0.0 for unvoiced frames.
    """
    # Try librosa first (faster)
    try:
        import librosa
        f0, voiced_flag, _ = librosa.pyin(
            waveform, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop,
            fill_na=0.0
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        return f0.astype(np.float32)
    except ImportError:
        pass

    # Pure-numpy YIN fallback
    n_frames = (len(waveform) - FRAME_LEN) // hop + 1
    f0 = np.zeros(n_frames, dtype=np.float32)
    tau_min = max(1, int(sr / fmax))
    tau_max = int(sr / fmin)

    for i in range(n_frames):
        start = i * hop
        frame = waveform[start: start + FRAME_LEN]
        if len(frame) < FRAME_LEN:
            break
        # Difference function
        d = np.zeros(tau_max)
        for tau in range(1, tau_max):
            diff = frame[:FRAME_LEN - tau] - frame[tau:FRAME_LEN]
            d[tau] = np.sum(diff ** 2)
        # Cumulative mean normalized difference
        cmnd = np.zeros(tau_max)
        cmnd[0] = 1.0
        running_sum = 0.0
        for tau in range(1, tau_max):
            running_sum += d[tau]
            cmnd[tau] = d[tau] * tau / (running_sum + 1e-8)
        # Find first minimum below threshold
        for tau in range(tau_min, tau_max - 1):
            if cmnd[tau] < threshold and cmnd[tau] < cmnd[tau + 1]:
                f0[i] = sr / tau
                break
    return f0


# ── Energy Extraction ─────────────────────────────────────────────────────────

def extract_energy(waveform: np.ndarray, hop: int = HOP_LENGTH,
                   frame_len: int = FRAME_LEN) -> np.ndarray:
    """RMS energy per frame (log scale)."""
    n_frames = (len(waveform) - frame_len) // hop + 1
    energy = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = waveform[start: start + frame_len]
        energy[i] = np.sqrt(np.mean(frame ** 2) + 1e-8)
    return np.log(energy + 1e-8).astype(np.float32)


# ── DTW ───────────────────────────────────────────────────────────────────────

def dtw(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Classic DTW between 1-D sequences x (ref) and y (synth).

    Returns
    -------
    path  : (2, K) array of (ref_idx, synth_idx) pairs
    cost  : total normalized distance
    """
    N, M = len(x), len(y)
    D = np.zeros((N + 1, M + 1), dtype=np.float64)
    D[0, :] = np.inf
    D[:, 0] = np.inf
    D[0, 0] = 0.0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = abs(float(x[i - 1]) - float(y[j - 1]))
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # Backtrack
    i, j = N, M
    path_i, path_j = [i - 1], [j - 1]
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            m = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
            if m == 0:
                i -= 1
            elif m == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path_i.append(i - 1)
        path_j.append(j - 1)

    path = np.array([path_i[::-1], path_j[::-1]])
    cost = D[N, M] / (N + M)
    return path, cost


def warp_contour(source: np.ndarray, path: np.ndarray, target_len: int) -> np.ndarray:
    """
    Warp a contour (source) using DTW path to match target_len.
    Returns interpolated 1-D array of length target_len.
    """
    # Map source frames to target positions via path
    src_idx = path[0]
    tgt_idx = path[1]

    # Interpolate source values at target positions
    interp = scipy.interpolate.interp1d(
        tgt_idx, source[src_idx],
        kind="linear", bounds_error=False,
        fill_value=(source[src_idx[0]], source[src_idx[-1]])
    )
    return interp(np.arange(target_len)).astype(np.float32)


# ── Pitch Shifting ────────────────────────────────────────────────────────────

def shift_pitch_frame(
    waveform: np.ndarray,
    f0_original: np.ndarray,
    f0_target: np.ndarray,
    sr: int,
    hop: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Frame-level pitch shifting using resampling trick (PSOLA-lite).
    For each voiced frame, compute shift ratio and resample.
    """
    out = waveform.copy()
    n_frames = min(len(f0_original), len(f0_target))

    for i in range(n_frames):
        f_orig = f0_original[i]
        f_tgt  = f0_target[i]

        # Only shift voiced frames
        if f_orig > 0 and f_tgt > 0:
            ratio = f_tgt / f_orig
            start = i * hop
            end   = start + hop
            if end > len(waveform):
                break
            frame = waveform[start:end]
            new_len = max(1, int(len(frame) / ratio))
            # Resample frame to shift pitch
            resampled = scipy.signal.resample(frame, new_len)
            # Fit back into original length
            fit_len = min(len(resampled), hop)
            out[start: start + fit_len] = resampled[:fit_len]

    return out


# ── Main Prosody Warping Pipeline ─────────────────────────────────────────────

def warp_prosody(
    professor_audio: str,
    synthesized_audio: str,
    output_audio: str,
    sr: int = SR,
) -> str:
    """
    Apply prosody warping: map professor's F0 + Energy onto synthesized audio.

    Parameters
    ----------
    professor_audio  : path to original lecture WAV
    synthesized_audio: path to raw TTS output WAV
    output_audio     : path to save prosody-warped WAV
    sr               : sample rate

    Returns
    -------
    output_audio path
    """
    print("[Prosody] Loading audio files ...")
    ref_wav,  ref_sr  = torchaudio.load(professor_audio)
    syn_wav,  syn_sr  = torchaudio.load(synthesized_audio)

    ref_np = ref_wav.mean(0).numpy()
    syn_np = syn_wav.mean(0).numpy()

    # Resample if needed
    if ref_sr != sr:
        ref_np = scipy.signal.resample(ref_np, int(len(ref_np) * sr / ref_sr))
    if syn_sr != sr:
        syn_np = scipy.signal.resample(syn_np, int(len(syn_np) * sr / syn_sr))

    print("[Prosody] Extracting F0 and Energy ...")
    ref_f0  = extract_f0_yin(ref_np, sr)
    syn_f0  = extract_f0_yin(syn_np, sr)
    ref_eng = extract_energy(ref_np)
    syn_eng = extract_energy(syn_np)

    print(f"[Prosody] Ref F0 frames: {len(ref_f0)}, Synth F0 frames: {len(syn_f0)}")

    # DTW on F0 (voiced frames only)
    ref_voiced = ref_f0[ref_f0 > 0]
    syn_voiced = syn_f0[syn_f0 > 0]

    if len(ref_voiced) > 0 and len(syn_voiced) > 0:
        print("[Prosody] Running DTW on F0 contours ...")
        path, dtw_cost = dtw(ref_voiced, syn_voiced)
        print(f"[Prosody] DTW cost = {dtw_cost:.4f}")

        # Warp reference F0 to match synthesized length
        warped_f0 = warp_contour(ref_voiced, path, len(syn_f0))
        # Apply pitch shift
        print("[Prosody] Applying pitch shifting ...")
        warped_audio = shift_pitch_frame(syn_np, syn_f0, warped_f0, sr)
    else:
        print("[Prosody] WARNING: not enough voiced frames, skipping pitch warp")
        warped_audio = syn_np

    # Save output
    out_tensor = torch.from_numpy(warped_audio).unsqueeze(0)
    os.makedirs(os.path.dirname(output_audio) or ".", exist_ok=True)
    torchaudio.save(output_audio, out_tensor, sr)
    print(f"[Prosody] Saved warped audio → {output_audio}")

    # Save F0 analysis for report
    analysis = {
        "ref_f0_mean_hz":  float(ref_f0[ref_f0 > 0].mean()) if (ref_f0 > 0).any() else 0,
        "syn_f0_mean_hz":  float(syn_f0[syn_f0 > 0].mean()) if (syn_f0 > 0).any() else 0,
        "dtw_cost":        float(dtw_cost) if len(ref_voiced) > 0 else None,
        "ref_energy_mean": float(ref_eng.mean()),
        "syn_energy_mean": float(syn_eng.mean()),
    }
    import json
    with open("output/prosody_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"[Prosody] Analysis: {analysis}")
    return output_audio


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--reference",   required=True, help="Professor's lecture WAV")
    p.add_argument("--synthesized", required=True, help="Raw TTS output WAV")
    p.add_argument("--output",      default="output/prosody_warped.wav")
    args = p.parse_args()
    warp_prosody(args.reference, args.synthesized, args.output)
