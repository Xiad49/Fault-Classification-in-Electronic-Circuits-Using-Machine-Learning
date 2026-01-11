"""
feature_extraction.py

Feature extraction for RC, RL, RLC circuits.
Supports:
1) CSV-based features
2) Raw simulation data (optional)

Project root: C:\\Project\\Code
"""

import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks, windows
from scipy.fft import rfft, rfftfreq


# ============================================================
# Optional simulation loader (SAFE)
# ============================================================

try:
    from simulate_circuits import load_simulation
except Exception:
        load_simulation = None
    

# ============================================================
# CSV MODE
# ============================================================

def extract_features_from_csv(
    csv_path: str,
    label_col: str = "fault_label"
) -> Tuple[pd.DataFrame, List[str]]:

    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError("CSV must contain 'fault_label' column")

    feature_cols = [
        c for c in df.columns
        if c != label_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feature_cols:
        raise ValueError("No numeric features found in CSV")

    return df, feature_cols


# ============================================================
# SIGNAL HELPERS
# ============================================================

def compute_basic_stats(x: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_range": float(np.ptp(x)),
    }


def compute_fft_features(
    t: np.ndarray,
    y: np.ndarray,
    prefix: str,
    n_peaks: int = 3
) -> Dict[str, float]:

    if len(t) < 2:
        return {}

    dt = t[1] - t[0]
    yf = np.abs(rfft(y * windows.hann(len(y))))
    xf = rfftfreq(len(y), dt)

    peaks, _ = find_peaks(yf)
    mags = yf[peaks]
    order = np.argsort(mags)[::-1]

    feats = {}
    for i in range(n_peaks):
        if i < len(order):
            idx = peaks[order[i]]
            feats[f"{prefix}_fft_{i+1}_freq"] = float(xf[idx])
            feats[f"{prefix}_fft_{i+1}_mag"] = float(yf[idx])
        else:
            feats[f"{prefix}_fft_{i+1}_freq"] = np.nan
            feats[f"{prefix}_fft_{i+1}_mag"] = np.nan

    return feats


def compute_psd_bandpower(
    t: np.ndarray,
    y: np.ndarray,
    bands: List[Tuple[float, float]]
) -> Dict[str, float]:

    if len(t) < 2:
        return {}

    fs = 1.0 / (t[1] - t[0])
    f, psd = welch(y, fs=fs)

    out = {}
    for i, (f1, f2) in enumerate(bands, 1):
        mask = (f >= f1) & (f <= f2)
        out[f"psd_band_{i}"] = float(np.trapz(psd[mask], f[mask])) if np.any(mask) else 0.0

    return out


# ============================================================
# SIMULATION MODE
# ============================================================

def extract_features_from_simulation(
    sim_data: Dict,
    circuit_type: str,
    fault_label: str,
    sim_id: str,
    psd_bands: Optional[List[Tuple[float, float]]] = None
) -> Dict[str, float]:

    feats: Dict[str, float] = {
        "circuit_type": circuit_type,
        "fault_label": fault_label,
        "sim_id": sim_id,
    }

    t = np.asarray(sim_data.get("time", []))
    v = np.asarray(sim_data.get("voltage", []))

    if t.size > 1 and v.size > 1:
        feats.update(compute_basic_stats(v, "v"))
        feats.update(compute_fft_features(t, v, "v"))

        if psd_bands:
            feats.update(compute_psd_bandpower(t, v, psd_bands))

    for k, v in sim_data.get("params", {}).items():
        feats[f"param_{k}"] = float(v)

    return feats


def build_feature_dataset(
    sim_paths: List[str],
    circuit_types: List[str],
    fault_labels: List[str],
    psd_bands: Optional[List[Tuple[float, float]]] = None
) -> pd.DataFrame:

    if load_simulation is None:
        raise RuntimeError("Simulation support unavailable (PySpice issue)")

    rows = []
    for p, c, f in zip(sim_paths, circuit_types, fault_labels):
        sim_data = load_simulation(p)
        rows.append(
            extract_features_from_simulation(
                sim_data,
                c,
                f,
                os.path.basename(p),
                psd_bands
            )
        )

    return pd.DataFrame(rows)
