"""MIT-licensed eye-tracking fixation detectors for OFAT robustness benchmarking.

Implements two complementary velocity-based fixation parsers from published descriptions:
1. Adaptive Slow/Fast Velocity Classifier (based on Hessels et al., 2020):
   Iterative sliding-window velocity threshold (mean + lambda * sd), minimum fixation
   duration filter, and slow-phase merging across sub-threshold fast segments.
2. Velocity-Threshold Identification / I-VT (Salvucci & Goldberg, 2000):
   Fixed angular-velocity threshold in degrees of visual angle per second (deg/s),
   short-gap merging, and minimum fixation duration filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

# NEMO Science Museum display geometry (Strauch et al., 2023)
SCREEN_W_PX = 1920.0
SCREEN_H_PX = 1080.0
DISP_W_MM = 598.0
DISP_H_MM = 336.0
VIEW_DIST_MM = 800.0
MM_PER_PX = 0.5 * (DISP_W_MM / SCREEN_W_PX + DISP_H_MM / SCREEN_H_PX)


def px_to_deg(px: np.ndarray | float) -> np.ndarray | float:
    """Convert Euclidean screen distance in pixels to degrees of visual angle (dva)."""
    mm = np.asarray(px, dtype=float) * MM_PER_PX
    return np.degrees(2.0 * np.arctan2(mm, 2.0 * VIEW_DIST_MM))


def compute_gaze_velocity(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Central-difference gaze speed in pixels/second (NaN where samples or dt are invalid)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(t)
    vel = np.full(n, np.nan, dtype=float)
    if n < 3:
        return vel

    dt1 = t[1:-1] - t[:-2]
    dt2 = t[2:] - t[1:-1]
    valid_dt = (dt1 > 1e-6) & (dt2 > 1e-6)

    vx = np.full(n - 2, np.nan, dtype=float)
    vy = np.full(n - 2, np.nan, dtype=float)
    vx[valid_dt] = 0.5 * (
        (x[1:-1][valid_dt] - x[:-2][valid_dt]) / dt1[valid_dt]
        + (x[2:][valid_dt] - x[1:-1][valid_dt]) / dt2[valid_dt]
    )
    vy[valid_dt] = 0.5 * (
        (y[1:-1][valid_dt] - y[:-2][valid_dt]) / dt1[valid_dt]
        + (y[2:][valid_dt] - y[1:-1][valid_dt]) / dt2[valid_dt]
    )
    vel[1:-1] = np.hypot(vx, vy)
    return vel


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive-exclusive (start, end) index pairs for True runs in a boolean mask."""
    if len(mask) == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _adaptive_window_threshold(
    vel_slice: np.ndarray,
    lam: float,
    max_iter: int = 50,
    init_thr: float = 1e12,
) -> float:
    """Iterative sigma-clipped velocity threshold (mean + lam * sd) on a local window."""
    v = vel_slice[np.isfinite(vel_slice)]
    if len(v) < 5:
        return np.nan
    thr = float(init_thr)
    for _ in range(max_iter):
        sub = v[v < thr]
        if len(sub) < 5:
            break
        mu = float(np.mean(sub))
        sd = float(np.std(sub, ddof=1))
        new_thr = mu + lam * sd
        if not np.isfinite(new_thr) or abs(new_thr - thr) <= 1.0:
            thr = new_thr
            break
        thr = new_thr
    return thr


def _merge_and_filter_fixations(
    runs: list[tuple[int, int]],
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    min_fix_s: float,
    min_amp_deg: float,
    max_merge_gap_s: float = 0.075,
) -> pd.DataFrame:
    """Merge adjacent slow phases separated by small spatial amplitude and filter by duration."""
    if not runs:
        return pd.DataFrame(columns=["onset", "offset", "duration", "avg_x", "avg_y", "Order"])

    # Drop runs that have no finite gaze samples
    clean_runs = []
    for s, e in runs:
        if np.any(np.isfinite(x[s:e])) and np.any(np.isfinite(y[s:e])):
            clean_runs.append([s, e])
    if not clean_runs:
        return pd.DataFrame(columns=["onset", "offset", "duration", "avg_x", "avg_y", "Order"])

    # Iteratively merge adjacent candidate fixations when inter-fixation shift < min_amp_deg
    # and gap is brief (prevents micro-saccade / 1-sample noise fragmentation)
    merged: list[list[int]] = [clean_runs[0]]
    for s, e in clean_runs[1:]:
        ps, pe = merged[-1]
        gap_s = float(t[s] - t[pe - 1])
        px_prev = float(np.nanmean(x[ps:pe]))
        py_prev = float(np.nanmean(y[ps:pe]))
        px_curr = float(np.nanmean(x[s:e]))
        py_curr = float(np.nanmean(y[s:e]))
        amp_deg = float(px_to_deg(np.hypot(px_curr - px_prev, py_curr - py_prev)))
        if gap_s <= max_merge_gap_s and amp_deg < min_amp_deg:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    rows = []
    for s, e in merged:
        end_idx = min(e, len(t) - 1)
        onset = float(t[s])
        offset = float(t[end_idx])
        dur = offset - onset
        if dur >= min_fix_s:
            rows.append(
                {
                    "onset": round(onset, 4),
                    "offset": round(offset, 4),
                    "duration": round(dur, 4),
                    "avg_x": round(float(np.nanmean(x[s:e])), 2),
                    "avg_y": round(float(np.nanmean(y[s:e])), 2),
                }
            )

    out = pd.DataFrame(rows)
    if len(out):
        out["Order"] = np.arange(len(out), dtype=int)
    else:
        out = pd.DataFrame(columns=["onset", "offset", "duration", "avg_x", "avg_y", "Order"])
    return out


def detect_fixations_adaptive(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    lam: float = 2.5,
    min_fix_s: float = 0.06,
    min_amp_deg: float = 1.0,
    window_samples: int = 480,
    step_samples: int = 12,
) -> pd.DataFrame:
    """Adaptive slow/fast phase velocity classifier (Hessels et al., 2020 formulation).

    Computes local sigma-clipped velocity thresholds over sliding windows and averages
    overlapping window thresholds across each sample before slow-phase merging.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(t)
    if n < 10:
        return pd.DataFrame(columns=["onset", "offset", "duration", "avg_x", "avg_y", "Order"])

    vel = compute_gaze_velocity(x, y, t)
    thr_acc = np.zeros(n, dtype=float)
    thr_cnt = np.zeros(n, dtype=float)

    if n <= window_samples:
        w_thr = _adaptive_window_threshold(vel, lam=lam)
        if np.isfinite(w_thr):
            thr_acc[:] = w_thr
            thr_cnt[:] = 1.0
    else:
        starts = list(range(0, n - window_samples, step_samples))
        if starts[-1] != n - window_samples:
            starts.append(n - window_samples)
        for st in starts:
            en = st + window_samples
            w_thr = _adaptive_window_threshold(vel[st:en], lam=lam)
            if np.isfinite(w_thr):
                thr_acc[st:en] += w_thr
                thr_cnt[st:en] += 1.0

    valid_thr = thr_cnt > 0
    if not np.any(valid_thr):
        return pd.DataFrame(columns=["onset", "offset", "duration", "avg_x", "avg_y", "Order"])

    thr = np.full(n, np.nan, dtype=float)
    thr[valid_thr] = thr_acc[valid_thr] / thr_cnt[valid_thr]

    is_slow = np.isfinite(vel) & np.isfinite(thr) & (vel <= thr) & np.isfinite(x) & np.isfinite(y)
    runs = _contiguous_true_runs(is_slow)
    return _merge_and_filter_fixations(
        runs, x, y, t, min_fix_s=min_fix_s, min_amp_deg=min_amp_deg
    )


def detect_fixations_ivt(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    vel_thr_deg_s: float = 30.0,
    min_fix_s: float = 0.06,
    min_amp_deg: float = 1.0,
) -> pd.DataFrame:
    """Classic I-VT fixation detector (Salvucci & Goldberg, 2000) in deg/s."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(t) < 10:
        return pd.DataFrame(columns=["onset", "offset", "duration", "avg_x", "avg_y", "Order"])

    vel_deg = px_to_deg(compute_gaze_velocity(x, y, t))
    is_slow = np.isfinite(vel_deg) & (vel_deg <= vel_thr_deg_s) & np.isfinite(x) & np.isfinite(y)
    runs = _contiguous_true_runs(is_slow)
    return _merge_and_filter_fixations(
        runs, x, y, t, min_fix_s=min_fix_s, min_amp_deg=min_amp_deg
    )
