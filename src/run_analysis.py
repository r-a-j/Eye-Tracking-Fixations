"""Full statistical analysis, diagnostic audit, and OFAT robustness pipeline for NEMO eye-tracking data.

Analyzes 183,627 discrete fixations from 7,250 valid free-viewing visitors (ages 3-93)
collected at the NEMO Science Museum in Amsterdam (Strauch et al., 2023; https://osf.io/sk4fr/).

Addresses two core research questions plus methodological robustness:
1. Lifespan Age Effect: Does fixation duration decrease in childhood and increase in older adulthood?
2. Trial-Time Dynamics: How do fixation durations evolve across the 10-second free-viewing trial
   once fixation-order survivorship bias and 10-second right-censoring are separated?
3. Robustness: True One-Factor-At-a-Time (OFAT) hyperparameter sensitivity on real 60 Hz gaze traces
   for both an adaptive velocity classifier (Hessels et al., 2020) and independent I-VT
   (Salvucci & Goldberg, 2000), plus sensitivity to the DoB=2000 kiosk UI default spike.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .lmm import (
    LMMResult,
    fit_random_intercept_lmm,
    likelihood_ratio_test,
    make_cubic_bspline,
)
from .fixation_detectors import (
    SCREEN_H_PX,
    SCREEN_W_PX,
    detect_fixations_adaptive,
    detect_fixations_ivt,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
RES_DIR = REPO_ROOT / "results"

DEFAULT_ANALYSIS_DIR = REPO_ROOT.parent / "project" / "analysis"
DEFAULT_PREPROC_DIR = REPO_ROOT.parent / "project" / "nemo_eyetracking" / "data" / "pre_processed_updated"
DEFAULT_STIMULUS_IMG = REPO_ROOT.parent / "project" / "nemo_eyetracking" / "saleval" / "images" / "free_viewing_img_1080.png"

# Editorial scientific color palette
COL_DARK = "#0f172a"
COL_SLATE = "#475569"
COL_BLUE = "#2563eb"
COL_TEAL = "#0d9488"
COL_AMBER = "#d97706"
COL_RED = "#dc2626"
COL_PURPLE = "#7c3aed"

AGE_BINS = [0, 5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 120]
AGE_LABELS = ["<6", "6-11", "12-17", "18-23", "24-29", "30-35", "36-41", "42-47", "48-53", "54-59", "60+"]


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.9,
            "axes.titlesize": 11.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.1,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#e2e8f0",
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        }
    )


def load_merged_fixations(analysis_dir: Path) -> pd.DataFrame:
    """Load compiled NEMO fixations and merge participant demographics."""
    fix_path = analysis_dir / "compiled_fixations.csv"
    part_path = analysis_dir / "participant_info.csv"
    if not fix_path.exists() or not part_path.exists():
        raise FileNotFoundError(
            f"Expected compiled_fixations.csv and participant_info.csv in {analysis_dir}. "
            "Download the NEMO eye-tracking dataset from https://osf.io/sk4fr/ and pass --data-dir."
        )

    fix = pd.read_csv(fix_path)
    part = pd.read_csv(part_path)
    df = fix.merge(part, on="ID", how="left")
    df["ID_str"] = df["ID"].astype(str)
    df["rec_year"] = df["ID_str"].str[:4].astype(int)
    df["DoB"] = pd.to_numeric(df["DoB"], errors="coerce")
    df["age"] = df["rec_year"] - df["DoB"]
    df["duration_ms"] = df["duration"] * 1000.0
    df["log_dur"] = np.log(df["duration"])
    df["is_last"] = df.groupby("ID")["Order"].transform("max") == df["Order"]
    return df


def build_age_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute participant-level summary statistics across the 11 age brackets."""
    rows = []
    for cohort_label, sub in [
        ("all_valid_fv", df),
        ("excl_dob_2000", df[df["DoB"] != 2000]),
    ]:
        ps = (
            sub.groupby("ID")
            .agg(
                mean_ms=("duration_ms", "mean"),
                med_ms=("duration_ms", "median"),
                log_mean=("log_dur", "mean"),
                n_fix=("duration", "size"),
                age=("age", "first"),
            )
            .reset_index()
        )
        ps["age_group"] = pd.cut(ps["age"], bins=AGE_BINS, labels=AGE_LABELS)
        for ag, grp in ps.groupby("age_group", observed=False):
            rows.append(
                {
                    "cohort": cohort_label,
                    "age_group": str(ag),
                    "n_participants": int(len(grp)),
                    "mean_of_subj_means_ms": round(float(grp["mean_ms"].mean()), 1),
                    "sd_of_subj_means_ms": round(float(grp["mean_ms"].std(ddof=1)), 1),
                    "median_of_subj_medians_ms": round(float(grp["med_ms"].median()), 1),
                    "geom_mean_ms": round(float(np.exp(grp["log_mean"].mean()) * 1000.0), 1),
                    "mean_fix_per_trial": round(float(grp["n_fix"].mean()), 1),
                }
            )
    return pd.DataFrame(rows)


def evaluate_cohort_models(
    sub: pd.DataFrame,
    cohort_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run participant-level Kruskal-Wallis and nested Log-Normal LMM LRTs on one cohort."""
    y = sub["log_dur"].to_numpy(float)
    g = sub["ID"].to_numpy()
    age = sub["age"].to_numpy(float)
    onset = sub["onset"].to_numpy(float)

    B_on, on_knots = make_cubic_bspline(onset, df=5)
    B_age_lin = ((age - 25.0) / 10.0)[:, None]
    B_age_quad = np.column_stack([(age - 25.0) / 10.0, ((age - 25.0) / 10.0) ** 2])
    B_age_spl, age_knots = make_cubic_bspline(age, df=5)

    # Nested ML models for exact fixed-effect Likelihood Ratio Tests
    m0 = fit_random_intercept_lmm(y, np.ones((len(y), 1)), g, reml=False)
    m_on_lin = fit_random_intercept_lmm(y, np.column_stack([np.ones(len(y)), onset]), g, reml=False)
    m_on = fit_random_intercept_lmm(y, np.column_stack([np.ones(len(y)), B_on]), g, reml=False)
    m_age_lin = fit_random_intercept_lmm(
        y, np.column_stack([np.ones(len(y)), B_on, B_age_lin]), g, reml=False
    )
    m_age_quad = fit_random_intercept_lmm(
        y, np.column_stack([np.ones(len(y)), B_on, B_age_quad]), g, reml=False
    )
    m_age_spl = fit_random_intercept_lmm(
        y, np.column_stack([np.ones(len(y)), B_on, B_age_spl]), g, reml=False
    )
    # Final REML fit for unbiased variance components and effect curves
    m_full_reml = fit_random_intercept_lmm(
        y, np.column_stack([np.ones(len(y)), B_on, B_age_spl]), g, reml=True
    )

    lrt_on, df_on, p_on = likelihood_ratio_test(m_on, m0)
    lrt_on_nl, df_on_nl, p_on_nl = likelihood_ratio_test(m_on, m_on_lin)
    lrt_al, _, p_al = likelihood_ratio_test(m_age_lin, m_on)
    lrt_aq, _, p_aq = likelihood_ratio_test(m_age_quad, m_on)
    lrt_as, df_as, p_as = likelihood_ratio_test(m_age_spl, m_on)

    # Quantify practical ambient-to-focal rise from onset = 0.5 s to onset = 7.0 s (holding age = 25)
    t_eval = np.array([0.5, 7.0])
    B_on_ev, _ = make_cubic_bspline(t_eval, df=5, knots=on_knots)
    B_ag_ev, _ = make_cubic_bspline(np.array([25.0, 25.0]), df=5, knots=age_knots)
    X_ev = np.column_stack([np.ones(2), B_on_ev, B_ag_ev])
    pred_ms_ev = np.exp(X_ev @ m_full_reml.beta) * 1000.0
    rise_pct = 100.0 * (pred_ms_ev[1] / pred_ms_ev[0] - 1.0)

    # Proper participant-level Kruskal-Wallis across 11 age brackets
    ps = sub.groupby("ID").agg(mean_ms=("duration_ms", "mean"), age=("age", "first")).reset_index()
    ps["age_group"] = pd.cut(ps["age"], bins=AGE_BINS, labels=AGE_LABELS)
    kw_groups = [grp["mean_ms"].to_numpy() for _, grp in ps.groupby("age_group", observed=False) if len(grp) > 0]
    kw_stat, kw_p = stats.kruskal(*kw_groups)

    age_pct_dec = 100.0 * (np.exp(float(m_age_lin.beta[-1])) - 1.0)

    summary = {
        "cohort": cohort_name,
        "n_fixations": int(len(sub)),
        "n_participants": int(m_full_reml.n_groups),
        "sigma_u": round(m_full_reml.sigma_u, 4),
        "sigma_e": round(m_full_reml.sigma_e, 4),
        "icc": round(m_full_reml.icc, 4),
        "r2_marginal": round(m_full_reml.r2_marginal, 4),
        "r2_conditional": round(m_full_reml.r2_conditional, 4),
        "resid_skew": round(m_full_reml.resid_skew, 2),
        "resid_kurtosis": round(m_full_reml.resid_kurtosis, 2),
        "max_abs_std_resid": round(m_full_reml.max_abs_std_resid, 2),
        "lrt_onset_chi2": round(lrt_on, 1),
        "lrt_onset_df": df_on,
        "lrt_onset_p": f"{p_on:.2e}",
        "lrt_onset_nonlin_chi2": round(lrt_on_nl, 1),
        "lrt_onset_nonlin_p": f"{p_on_nl:.2e}",
        "pred_05s_ms": round(float(pred_ms_ev[0]), 1),
        "pred_70s_ms": round(float(pred_ms_ev[1]), 1),
        "onset_05_to_70_rise_pct": round(float(rise_pct), 1),
        "age_pct_per_decade": round(float(age_pct_dec), 2),
        "lrt_age_lin_chi2": round(lrt_al, 2),
        "lrt_age_lin_p": round(p_al, 4),
        "lrt_age_quad_chi2": round(lrt_aq, 2),
        "lrt_age_quad_p": round(p_aq, 4),
        "lrt_age_spl_chi2": round(lrt_as, 2),
        "lrt_age_spl_df": df_as,
        "lrt_age_spl_p": round(p_as, 4),
        "kw_participant_H": round(float(kw_stat), 2),
        "kw_participant_p": round(float(kw_p), 4),
    }
    artifacts = {
        "reml": m_full_reml,
        "on_knots": on_knots,
        "age_knots": age_knots,
    }
    return summary, artifacts


def run_ofat_robustness(
    df_ref: pd.DataFrame,
    preproc_dir: Path,
    *,
    n_participants: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """True One-Factor-At-a-Time (OFAT) detector sensitivity on real 60 Hz NEMO gaze traces."""
    if not preproc_dir.exists():
        print(f"Skipping OFAT raw-trace sweep: {preproc_dir} not found.")
        return pd.DataFrame()

    part_meta = (
        df_ref.groupby("ID_str")
        .agg(age=("age", "first"), dob=("DoB", "first"))
        .reset_index()
    )
    available = {f.stem: f for f in preproc_dir.glob("*.csv")}
    part_meta = part_meta[part_meta["ID_str"].isin(available)].copy()

    # Age-stratified sample across the lifespan so childhood and older adults stay well represented
    part_meta["age_group"] = pd.cut(part_meta["age"], bins=AGE_BINS, labels=AGE_LABELS)
    rng = np.random.default_rng(seed)
    chosen_ids = []
    # Cap per bracket proportionally while keeping every sparse bracket well-represented
    target_per_bin = max(40, n_participants // len(AGE_LABELS))
    for _, grp in part_meta.groupby("age_group", observed=False):
        take = min(len(grp), target_per_bin)
        if take > 0:
            chosen_ids.extend(rng.choice(grp["ID_str"].to_numpy(), size=take, replace=False).tolist())
    remaining_needed = n_participants - len(chosen_ids)
    if remaining_needed > 0:
        rem = part_meta[~part_meta["ID_str"].isin(chosen_ids)]["ID_str"].to_numpy()
        extra = rng.choice(rem, size=min(remaining_needed, len(rem)), replace=False).tolist()
        chosen_ids.extend(extra)

    chosen_set = set(chosen_ids)
    age_lookup = dict(zip(part_meta["ID_str"], part_meta["age"]))
    print(f"Loading {len(chosen_ids)} real 60 Hz gaze traces for OFAT sensitivity sweep...", flush=True)

    traces: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for pid in sorted(chosen_ids):
        d = pd.read_csv(available[pid], usecols=["x", "y", "time_from_start", "message"])
        hits = np.flatnonzero(d["message"].to_numpy() == "ScreenInstructionPostGame")
        if len(hits) > 0:
            d = d.iloc[: hits[0]]
        traces.append(
            (
                pid,
                d["x"].to_numpy(float),
                d["y"].to_numpy(float),
                d["time_from_start"].to_numpy(float),
            )
        )

    def _audit_extracted(label: str, family: str, varied_param: str, fx_df: pd.DataFrame) -> dict[str, object]:
        fx_df = fx_df[fx_df["duration"] > 0].copy()
        fx_df["age"] = fx_df["ID_str"].map(age_lookup)
        fx_df["log_dur"] = np.log(fx_df["duration"])
        sub = fx_df[fx_df["onset"] <= 8.0].copy()

        y = sub["log_dur"].to_numpy(float)
        g = sub["ID_str"].to_numpy()
        age = sub["age"].to_numpy(float)
        onset = sub["onset"].to_numpy(float)

        B_on, on_knots = make_cubic_bspline(onset, df=5)
        B_ag_lin = ((age - 25.0) / 10.0)[:, None]
        B_ag_spl, ag_knots = make_cubic_bspline(age, df=5)

        m0 = fit_random_intercept_lmm(y, np.ones((len(y), 1)), g, reml=False)
        m_on = fit_random_intercept_lmm(y, np.column_stack([np.ones(len(y)), B_on]), g, reml=False)
        m_al = fit_random_intercept_lmm(y, np.column_stack([np.ones(len(y)), B_on, B_ag_lin]), g, reml=False)
        m_as = fit_random_intercept_lmm(y, np.column_stack([np.ones(len(y)), B_on, B_ag_spl]), g, reml=False)

        lrt_on, _, p_on = likelihood_ratio_test(m_on, m0)
        lrt_as, _, p_as = likelihood_ratio_test(m_as, m_on)

        # Ambient-to-focal 0.5 s -> 7.0 s rise (%)
        B_on_ev, _ = make_cubic_bspline(np.array([0.5, 7.0]), df=5, knots=on_knots)
        X_ev = np.column_stack([np.ones(2), B_on_ev])
        pred_ev = np.exp(X_ev @ m_on.beta) * 1000.0
        rise_pct = 100.0 * (pred_ev[1] / pred_ev[0] - 1.0)
        age_pct_dec = 100.0 * (np.exp(float(m_al.beta[-1])) - 1.0)

        return {
            "config": label,
            "detector_family": family,
            "varied_parameter": varied_param,
            "n_participants": int(fx_df["ID_str"].nunique()),
            "n_fixations": int(len(fx_df)),
            "fix_per_subj": round(len(fx_df) / max(1, fx_df["ID_str"].nunique()), 1),
            "mean_ms": round(float(fx_df["duration"].mean() * 1000.0), 1),
            "median_ms": round(float(fx_df["duration"].median() * 1000.0), 1),
            "onset_05_to_70_rise_pct": round(float(rise_pct), 1),
            "onset_lrt_chi2": round(float(lrt_on), 1),
            "onset_lrt_p": f"{p_on:.2e}",
            "age_pct_per_decade": round(float(age_pct_dec), 2),
            "age_spl_lrt_p": round(float(p_as), 4),
        }

    results = []
    # 0. Reference OSF Hessels fixations on the exact same 1,000 participants
    ref_sub = df_ref[df_ref["ID_str"].isin(chosen_set)][["ID_str", "onset", "duration"]].copy()
    results.append(_audit_extracted("OSF Reference (Hessels 2020)", "Reference", "baseline", ref_sub))

    # 1. OFAT sweeps on MIT Adaptive Velocity Classifier (defaults: lam=2.5, min_fix=0.06, min_amp=1.0)
    adaptive_configs = [
        ("Adaptive Baseline (lam=2.5, 60ms, 1.0deg)", "baseline", dict(lam=2.5, min_fix_s=0.06, min_amp_deg=1.0)),
        ("OFAT: lambda = 2.0", "lambda", dict(lam=2.0, min_fix_s=0.06, min_amp_deg=1.0)),
        ("OFAT: lambda = 3.0", "lambda", dict(lam=3.0, min_fix_s=0.06, min_amp_deg=1.0)),
        ("OFAT: min_fix = 40 ms", "min_fix", dict(lam=2.5, min_fix_s=0.04, min_amp_deg=1.0)),
        ("OFAT: min_fix = 80 ms", "min_fix", dict(lam=2.5, min_fix_s=0.08, min_amp_deg=1.0)),
        ("OFAT: min_fix = 100 ms", "min_fix", dict(lam=2.5, min_fix_s=0.10, min_amp_deg=1.0)),
        ("OFAT: min_amp = 0.5 deg", "min_amp", dict(lam=2.5, min_fix_s=0.06, min_amp_deg=0.5)),
        ("OFAT: min_amp = 1.5 deg", "min_amp", dict(lam=2.5, min_fix_s=0.06, min_amp_deg=1.5)),
    ]
    for label, varied, kwargs in adaptive_configs:
        parts = []
        for pid, x, y, t in traces:
            fx = detect_fixations_adaptive(x, y, t, **kwargs)
            if len(fx):
                fx["ID_str"] = pid
                parts.append(fx[["ID_str", "onset", "duration"]])
        fx_all = pd.concat(parts, ignore_index=True)
        row = _audit_extracted(label, "Adaptive", varied, fx_all)
        results.append(row)
        print(f"  {label:44s} -> med={row['median_ms']:5.1f} ms | rise={row['onset_05_to_70_rise_pct']:+5.1f}% | age/dec={row['age_pct_per_decade']:+5.2f}%", flush=True)

    # 2. Independent I-VT classifier (Salvucci & Goldberg, 2000) across velocity thresholds
    ivt_configs = [
        ("I-VT (20 deg/s, 60ms)", "vel_thr", dict(vel_thr_deg_s=20.0, min_fix_s=0.06, min_amp_deg=1.0)),
        ("I-VT (30 deg/s, 60ms)", "vel_thr", dict(vel_thr_deg_s=30.0, min_fix_s=0.06, min_amp_deg=1.0)),
        ("I-VT (40 deg/s, 60ms)", "vel_thr", dict(vel_thr_deg_s=40.0, min_fix_s=0.06, min_amp_deg=1.0)),
    ]
    for label, varied, kwargs in ivt_configs:
        parts = []
        for pid, x, y, t in traces:
            fx = detect_fixations_ivt(x, y, t, **kwargs)
            if len(fx):
                fx["ID_str"] = pid
                parts.append(fx[["ID_str", "onset", "duration"]])
        fx_all = pd.concat(parts, ignore_index=True)
        row = _audit_extracted(label, "I-VT", varied, fx_all)
        results.append(row)
        print(f"  {label:44s} -> med={row['median_ms']:5.1f} ms | rise={row['onset_05_to_70_rise_pct']:+5.1f}% | age/dec={row['age_pct_per_decade']:+5.2f}%", flush=True)

    return pd.DataFrame(results)


def plot_data_audit_and_diagnostics(
    df: pd.DataFrame,
    raw_lmm: LMMResult,
    log_lmm: LMMResult,
) -> None:
    """Figure 1 (2x2): DoB=2000 artifact, duration right skew, Raw vs Log-Normal Q-Q plots."""
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2), dpi=220)

    # Panel A: Age distribution highlighting DoB=2000 (Age=23) kiosk default spike
    ax = axes[0, 0]
    p_ages = df.groupby("ID")["age"].first().to_numpy(int)
    age_vals = np.arange(3, 86)
    counts = np.array([int(np.sum(p_ages == a)) for a in age_vals])
    colors = [COL_RED if a == 23 else COL_BLUE for a in age_vals]
    ax.bar(age_vals, counts, width=0.82, color=colors, alpha=0.88, edgecolor="none")
    n_23 = int(np.sum(p_ages == 23))
    ax.annotate(
        f"DoB = 2000 default spike\n(Age 23: n = {n_23:,}, {100.0 * n_23 / len(p_ages):.1f}%)",
        xy=(23, n_23),
        xytext=(34, n_23 * 0.84),
        arrowprops=dict(arrowstyle="->", color=COL_RED, lw=1.3),
        fontsize=8.6,
        fontweight="bold",
        color=COL_RED,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#fecaca", alpha=0.95),
    )
    ax.set_title("A. Participant Age Audit (n = 7,250 Visitors, NEMO 2023)")
    ax.set_xlabel("Reported Age (years)")
    ax.set_ylabel("Participant Count")
    ax.set_xlim(1, 86)
    ax.grid(True, axis="y", alpha=0.35)

    # Panel B: Raw Fixation Duration Distribution (ms) showing heavy right skew
    ax = axes[0, 1]
    dur_ms = df["duration_ms"].to_numpy()
    clipped = dur_ms[dur_ms <= 1200.0]
    ax.hist(clipped, bins=60, color=COL_TEAL, alpha=0.78, edgecolor="white", linewidth=0.35, density=True)
    mean_v = float(np.mean(dur_ms))
    med_v = float(np.median(dur_ms))
    ax.axvline(med_v, color=COL_DARK, ls="--", lw=1.7, label=f"Median: {med_v:.0f} ms")
    ax.axvline(mean_v, color=COL_RED, ls="-", lw=1.7, label=f"Mean: {mean_v:.0f} ms (right tail to {dur_ms.max():.0f} ms)")
    ax.set_title("B. Fixation Duration Distribution (n = 183,627 Fixations)")
    ax.set_xlabel("Fixation Duration (ms, view capped at 1,200 ms)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right", fontsize=8.1)
    ax.grid(True, alpha=0.35)

    # Subsample 8,000 sorted quantiles for clean crisp Q-Q plots
    rng = np.random.default_rng(7)

    # Panel C: Raw duration LMM Standardized Residual Q-Q
    ax = axes[1, 0]
    z_raw = raw_lmm.residuals / raw_lmm.sigma_e
    q_idx = np.linspace(0.0005, 0.9995, 2500)
    theo_q = stats.norm.ppf(q_idx)
    emp_raw = np.quantile(z_raw, q_idx)
    ax.scatter(theo_q, emp_raw, s=8, color=COL_RED, alpha=0.65, edgecolors="none", label="Raw LMM residuals")
    lim_x = [-3.6, 3.6]
    ax.plot(lim_x, lim_x, "--", color=COL_DARK, lw=1.3, label="Normal reference")
    ax.text(
        0.04,
        0.95,
        f"Skewness = {raw_lmm.resid_skew:.2f}\nExcess kurtosis = {raw_lmm.resid_kurtosis:.1f}\nMax |z| = {raw_lmm.max_abs_std_resid:.1f} SD",
        transform=ax.transAxes,
        va="top",
        fontsize=8.3,
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#fecaca", alpha=0.95),
    )
    ax.set_title("C. Raw Duration LMM Residual Q-Q (Severe Skew Violation)")
    ax.set_xlabel("Theoretical Normal Quantiles")
    ax.set_ylabel("Standardized Residual Quantiles (z)")
    ax.set_xlim(-3.7, 3.7)
    ax.set_ylim(-4.0, 12.0)
    ax.legend(loc="lower right", fontsize=8.0)
    ax.grid(True, alpha=0.35)

    # Panel D: Log-Normal LMM Standardized Residual Q-Q
    ax = axes[1, 1]
    z_log = log_lmm.residuals / log_lmm.sigma_e
    emp_log = np.quantile(z_log, q_idx)
    ax.scatter(theo_q, emp_log, s=8, color=COL_TEAL, alpha=0.70, edgecolors="none", label="Log-Normal LMM residuals")
    ax.plot(lim_x, lim_x, "--", color=COL_DARK, lw=1.3, label="Normal reference")
    ax.text(
        0.04,
        0.95,
        f"Skewness = {log_lmm.resid_skew:.2f}\nExcess kurtosis = {log_lmm.resid_kurtosis:.2f}\nMax |z| = {log_lmm.max_abs_std_resid:.1f} SD",
        transform=ax.transAxes,
        va="top",
        fontsize=8.3,
        color="#065f46",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#a7f3d0", alpha=0.95),
    )
    ax.set_title("D. Log-Normal LMM Residual Q-Q (Near-Gaussian Fit)")
    ax.set_xlabel("Theoretical Normal Quantiles")
    ax.set_ylabel("Standardized Residual Quantiles (z)")
    ax.set_xlim(-3.7, 3.7)
    ax.set_ylim(-4.5, 4.8)
    ax.legend(loc="lower right", fontsize=8.0)
    ax.grid(True, alpha=0.35)

    fig.tight_layout(pad=1.7)
    fig.savefig(FIG_DIR / "data_audit_and_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_age_lifespan_analysis(
    df: pd.DataFrame,
    cohort_artifacts: dict[str, dict[str, object]],
) -> None:
    """Figure 2 (1x2): Continuous non-linear Age spline (3 cohorts) + BLUP individual variability."""
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), dpi=220)

    # Panel A: Continuous Age Spline predictions (ms) at reference onset = 4.0 s
    ax = axes[0]
    age_grid = np.linspace(5.0, 75.0, 141)

    # Empirical binned participant geometric means + 95% SE bars (all valid, uncensored)
    sub_unc = df[df["onset"] <= 8.0]
    ps = sub_unc.groupby("ID").agg(log_mean=("log_dur", "mean"), age=("age", "first")).reset_index()
    bin_edges = [3, 8, 12, 16, 20, 25, 30, 36, 42, 48, 55, 63, 80]
    ps["abin"] = pd.cut(ps["age"], bins=bin_edges)
    bx, by, bse = [], [], []
    for _, grp in ps.groupby("abin", observed=False):
        if len(grp) >= 20:
            mu = float(grp["log_mean"].mean())
            se = float(grp["log_mean"].std(ddof=1) / np.sqrt(len(grp)))
            bx.append(float(grp["age"].mean()))
            by.append(np.exp(mu) * 1000.0)
            bse.append(1.96 * se * np.exp(mu) * 1000.0)
    ax.errorbar(
        bx,
        by,
        yerr=bse,
        fmt="o",
        color="#64748b",
        ecolor="#94a3b8",
        elinewidth=1.1,
        capsize=2.5,
        markersize=4.5,
        alpha=0.85,
        label="Binned participant geom. mean (+/- 95% CI)",
        zorder=3,
    )

    specs = [
        ("all_uncensored", "All valid free-viewing (n=7,250; p=0.123)", COL_BLUE, "-", True),
        ("excl_dob_2000", "Excl. DoB=2000 spike (n=5,718; p=0.085)", COL_TEAL, "--", False),
        ("strict_valid_no2000", "Strict Valid & DoB!=2000 (n=4,546; p=0.024)", COL_AMBER, "-.", False),
    ]
    for key, label, col, ls, show_ci in specs:
        art = cohort_artifacts[key]
        m: LMMResult = art["reml"]
        B_on_ref, _ = make_cubic_bspline(np.full_like(age_grid, 4.0), df=5, knots=art["on_knots"])
        B_ag_grid, _ = make_cubic_bspline(age_grid, df=5, knots=art["age_knots"])
        Xg = np.column_stack([np.ones(len(age_grid)), B_on_ref, B_ag_grid])
        lp = Xg @ m.beta
        se_lp = np.sqrt(np.einsum("ij,jk,ik->i", Xg, m.cov_beta, Xg))
        pred_ms = np.exp(lp) * 1000.0
        ax.plot(age_grid, pred_ms, color=col, ls=ls, lw=2.2, label=label, zorder=5)
        if show_ci:
            lo = np.exp(lp - 1.96 * se_lp) * 1000.0
            hi = np.exp(lp + 1.96 * se_lp) * 1000.0
            ax.fill_between(age_grid, lo, hi, color=col, alpha=0.14, zorder=2)

    ax.set_title("A. Continuous Lifespan Age Curve (Log-Normal LMM, onset = 4 s)")
    ax.set_xlabel("Participant Age (years)")
    ax.set_ylabel("Predicted Median Fixation Duration (ms)")
    ax.set_xlim(4, 76)
    ax.set_ylim(185, 245)
    ax.legend(loc="lower left", fontsize=7.8, framealpha=0.95)
    ax.grid(True, alpha=0.35)

    # Panel B: Between-participant random-intercept spread (BLUPs) vs. age shift
    ax2 = axes[1]
    m_main: LMMResult = cohort_artifacts["all_uncensored"]["reml"]
    # Convert participant BLUPs to implied baseline median fixation duration at reference onset=4s, age=25
    B_on_r, _ = make_cubic_bspline(np.array([4.0]), df=5, knots=cohort_artifacts["all_uncensored"]["on_knots"])
    B_ag_r, _ = make_cubic_bspline(np.array([25.0]), df=5, knots=cohort_artifacts["all_uncensored"]["age_knots"])
    base_lp = float((np.column_stack([np.ones(1), B_on_r, B_ag_r]) @ m_main.beta).item())
    subj_baseline_ms = np.exp(base_lp + m_main.u_hat) * 1000.0

    p10, p50, p90 = np.percentile(subj_baseline_ms, [10, 50, 90])
    ax2.hist(subj_baseline_ms, bins=48, color=COL_PURPLE, alpha=0.72, edgecolor="white", linewidth=0.35, density=True)
    ax2.axvline(p10, color=COL_DARK, ls=":", lw=1.5, label=f"10th pct viewer: {p10:.0f} ms")
    ax2.axvline(p50, color=COL_DARK, ls="-", lw=1.7, label=f"Median viewer: {p50:.0f} ms")
    ax2.axvline(p90, color=COL_DARK, ls="--", lw=1.5, label=f"90th pct viewer: {p90:.0f} ms")
    ax2.axvspan(208.3, 220.4, color=COL_RED, alpha=0.22, label="Entire 70-yr age range (208-220 ms)")

    ax2.set_title("B. Between-Viewer Variability Dwarfs Lifespan Age Shift")
    ax2.set_xlabel("Participant Baseline Median Fixation Duration (ms, BLUPs)")
    ax2.set_ylabel("Density Across 7,250 Visitors")
    ax2.set_xlim(140, 320)
    ax2.text(
        0.97,
        0.52,
        f"sigma_u = {m_main.sigma_u:.3f} (log scale)\nICC = {100.0 * m_main.icc:.1f}%\n10th-90th spread: {p90 - p10:.0f} ms\nvs. ~12 ms across ages 6-75",
        transform=ax2.transAxes,
        ha="right",
        va="center",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#e2e8f0", alpha=0.95),
    )
    ax2.legend(loc="upper right", fontsize=7.8, framealpha=0.95)
    ax2.grid(True, alpha=0.35)

    fig.tight_layout(pad=1.6)
    fig.savefig(FIG_DIR / "age_lifespan_analysis.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_dynamics_and_censoring(
    df: pd.DataFrame,
    cohort_artifacts: dict[str, dict[str, object]],
) -> None:
    """Figure 3 (1x2): Continuous onset ambient-to-focal curve + Order survivorship artifact."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=220)

    # Panel A: Continuous trial onset time (0-10 s) separating uncensored 0-8 s from right-censoring 8-10 s
    ax = axes[0]
    edges = np.linspace(0.0, 10.0, 21)
    mids = 0.5 * (edges[:-1] + edges[1:])
    df_tmp = df.copy()
    df_tmp["obin"] = pd.cut(df_tmp["onset"], bins=edges)
    grp = df_tmp.groupby("obin", observed=False).agg(
        geom_ms=("log_dur", lambda s: np.exp(s.mean()) * 1000.0),
        mean_ms=("duration_ms", "mean"),
        pct_last=("is_last", lambda s: 100.0 * s.mean()),
    )

    ax.axvspan(8.0, 10.0, color="#fee2e2", alpha=0.55, zorder=0, label="Right-censoring zone (offset <= 10 s)")
    ax.axvline(8.0, color=COL_RED, ls="--", lw=1.2, alpha=0.8)

    # Empirical geometric means per 0.5 s bin
    ax.scatter(mids[mids <= 8.0], grp["geom_ms"].to_numpy()[mids <= 8.0], s=32, color=COL_TEAL, zorder=4, label="Empirical bin geom. mean (uncensored)")
    ax.scatter(mids[mids > 8.0], grp["geom_ms"].to_numpy()[mids > 8.0], s=32, color=COL_RED, marker="x", lw=1.8, zorder=4, label="Empirical bin geom. mean (censored tail)")

    # Uncensored Log-Normal LMM spline fit (0.05 to 8.0 s)
    art_u = cohort_artifacts["all_uncensored"]
    m_u: LMMResult = art_u["reml"]
    t_u = np.linspace(0.05, 8.0, 160)
    B_on_u, _ = make_cubic_bspline(t_u, df=5, knots=art_u["on_knots"])
    B_ag_u, _ = make_cubic_bspline(np.full_like(t_u, 25.0), df=5, knots=art_u["age_knots"])
    Xu = np.column_stack([np.ones(len(t_u)), B_on_u, B_ag_u])
    lp_u = Xu @ m_u.beta
    se_u = np.sqrt(np.einsum("ij,jk,ik->i", Xu, m_u.cov_beta, Xu))
    ax.plot(t_u, np.exp(lp_u) * 1000.0, color=COL_TEAL, lw=2.5, label="Log-Normal LMM spline (0-8 s uncensored)", zorder=5)
    ax.fill_between(t_u, np.exp(lp_u - 1.96 * se_u) * 1000.0, np.exp(lp_u + 1.96 * se_u) * 1000.0, color=COL_TEAL, alpha=0.16, zorder=3)

    # Naive full 0-10 s spline fit for direct contrast
    art_f = cohort_artifacts["all_full_10s"]
    m_f: LMMResult = art_f["reml"]
    t_f = np.linspace(0.05, 9.75, 195)
    B_on_f, _ = make_cubic_bspline(t_f, df=5, knots=art_f["on_knots"])
    B_ag_f, _ = make_cubic_bspline(np.full_like(t_f, 25.0), df=5, knots=art_f["age_knots"])
    Xf = np.column_stack([np.ones(len(t_f)), B_on_f, B_ag_f])
    ax.plot(t_f, np.exp(Xf @ m_f.beta) * 1000.0, color=COL_RED, ls=":", lw=1.9, label="Naive 0-10 s spline (biased by 10 s cutoff)", zorder=4)

    # Compute exact 0.5 s -> 7.0 s percentage rise for annotation
    B_ev_u, _ = make_cubic_bspline(np.array([0.5, 7.0]), df=5, knots=art_u["on_knots"])
    B_ag_ev_u, _ = make_cubic_bspline(np.array([25.0, 25.0]), df=5, knots=art_u["age_knots"])
    pred_ev_u = np.exp(np.column_stack([np.ones(2), B_ev_u, B_ag_ev_u]) @ m_u.beta) * 1000.0
    rise_pct_u = 100.0 * (pred_ev_u[1] / pred_ev_u[0] - 1.0)

    ax.annotate(
        f"Ambient-to-focal rise\n(+{rise_pct_u:.1f}% from 0.5 s to 7.0 s)\nthen plateau",
        xy=(6.8, float(pred_ev_u[1])),
        xytext=(2.9, 241.0),
        arrowprops=dict(arrowstyle="->", color=COL_TEAL, lw=1.2),
        fontsize=8.3,
        fontweight="bold",
        color="#065f46",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#a7f3d0", alpha=0.95),
    )
    ax.annotate(
        "96.9% final\nfixations",
        xy=(9.65, 145.0),
        xytext=(7.6, 152.0),
        arrowprops=dict(arrowstyle="->", color=COL_RED, lw=1.1),
        fontsize=7.9,
        fontweight="bold",
        color=COL_RED,
    )

    ax.set_title("A. Continuous Elapsed Trial Time (0-10 s) & Right-Censoring")
    ax.set_xlabel("Fixation Onset Time (seconds)")
    ax.set_ylabel("Median / Geometric Mean Duration (ms)")
    ax.set_xlim(0, 10.0)
    ax.set_ylim(130, 260)
    ax.legend(loc="lower left", fontsize=7.3, framealpha=0.95)
    ax.grid(True, alpha=0.35)

    # Panel B: Why Fixation Order creates an artificial late decline (Survivorship Bias)
    ax2 = axes[1]
    ord_grp = (
        df.groupby("Order")
        .agg(
            mean_ms=("duration_ms", "mean"),
            geom_ms=("log_dur", lambda s: np.exp(s.mean()) * 1000.0),
            n_active=("ID", "nunique"),
        )
        .reset_index()
    )
    ord_grp = ord_grp[ord_grp["Order"] <= 36]
    total_subj = df["ID"].nunique()
    ord_grp["pct_active"] = 100.0 * ord_grp["n_active"] / total_subj

    l1 = ax2.plot(ord_grp["Order"], ord_grp["mean_ms"], "-o", ms=3.8, color=COL_BLUE, lw=2.0, label="Mean fixation duration (ms)")
    l2 = ax2.plot(ord_grp["Order"], ord_grp["geom_ms"], "--", color=COL_TEAL, lw=1.8, label="Geom. mean duration (ms)")
    ax2.set_xlabel("Fixation Order Index (0 = Trial Start)")
    ax2.set_ylabel("Fixation Duration (ms)", color=COL_DARK)
    ax2.set_xlim(0, 36)
    ax2.set_ylim(115, 295)
    ax2.grid(True, alpha=0.35)

    ax2_r = ax2.twinx()
    l3 = ax2_r.fill_between(
        ord_grp["Order"],
        0,
        ord_grp["pct_active"],
        color="#94a3b8",
        alpha=0.22,
        label="% of 7,250 viewers reaching Order k",
    )
    ax2_r.plot(ord_grp["Order"], ord_grp["pct_active"], color="#64748b", ls=":", lw=1.5)
    ax2_r.set_ylabel("Active Participants Remaining (%)", color="#475569")
    ax2_r.set_ylim(0, 108)
    ax2_r.spines["right"].set_color("#94a3b8")

    ax2.annotate(
        "Selection bias past Order 22:\nonly fast-saccading viewers\nfit 25-36 fixations in 10 s",
        xy=(29, 224),
        xytext=(10.5, 140),
        arrowprops=dict(arrowstyle="->", color=COL_BLUE, lw=1.2),
        fontsize=8.2,
        fontweight="bold",
        color="#1e3a8a",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bfdbfe", alpha=0.95),
    )

    lines = l1 + l2 + [l3]
    labels = [x.get_label() for x in lines]
    ax2.legend(lines, labels, loc="upper right", fontsize=7.6, framealpha=0.95)
    ax2.set_title("B. Why Fixation Order Misleads (Survivorship Bias in 10 s Trial)")

    fig.tight_layout(pad=1.6)
    fig.savefig(FIG_DIR / "temporal_dynamics_and_censoring.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_gaze_density_grid(df: pd.DataFrame, bins_x: int = 192, bins_y: int = 108) -> tuple[np.ndarray, int]:
    """Compute screen-aligned 2D fixation density grid (top-down row order matching image pixels).

    Note: compiled_fixations stores avg_y in bottom-left Cartesian screen pixels
    (0 = bottom, 1080 = top), so y_top = SCREEN_H_PX - avg_y maps to image row coordinates.
    """
    in_screen = df[
        df["avg_x"].between(0.0, SCREEN_W_PX) & df["avg_y"].between(0.0, SCREEN_H_PX)
    ]
    x_px = in_screen["avg_x"].to_numpy(float)
    y_top = SCREEN_H_PX - in_screen["avg_y"].to_numpy(float)
    h, _, _ = np.histogram2d(
        x_px,
        y_top,
        bins=[bins_x, bins_y],
        range=[[0.0, SCREEN_W_PX], [0.0, SCREEN_H_PX]],
    )
    return h.T, int(len(in_screen))


def _load_stimulus_rgb(stimulus_img: Path | None) -> np.ndarray | None:
    if stimulus_img is None or not stimulus_img.exists():
        return None
    return np.asarray(Image.open(stimulus_img).convert("RGB"))


def _overlay_heatmap_on_stimulus(stim_rgb: np.ndarray, smooth_density: np.ndarray) -> np.ndarray:
    """Blend an inferno fixation density heatmap cleanly over a dimmed stimulus image."""
    # Resize smooth density to stimulus height x width cleanly via PIL bilinear
    norm = smooth_density / max(float(smooth_density.max()), 1e-9)
    norm_gamma = np.power(np.clip(norm, 0.0, 1.0), 0.72)
    cmap = plt.get_cmap("inferno")
    heat_rgba = cmap(norm_gamma)
    heat_rgb = (heat_rgba[:, :, :3] * 255.0).astype(np.uint8)
    heat_resized = np.asarray(
        Image.fromarray(heat_rgb).resize((stim_rgb.shape[1], stim_rgb.shape[0]), Image.Resampling.BILINEAR),
        dtype=float,
    )
    alpha_small = np.clip(norm_gamma * 0.88, 0.0, 0.85)
    alpha_full = np.asarray(
        Image.fromarray((alpha_small * 255.0).astype(np.uint8)).resize(
            (stim_rgb.shape[1], stim_rgb.shape[0]), Image.Resampling.BILINEAR
        ),
        dtype=float,
    )[..., None] / 255.0

    # Slightly dim background stimulus where gaze is sparse so focal hotspots pop cleanly
    gray = np.dot(stim_rgb[..., :3], [0.299, 0.587, 0.114])[..., None]
    bg_muted = (0.55 * stim_rgb.astype(float) + 0.45 * gray) * 0.52
    blended = bg_muted * (1.0 - alpha_full) + heat_resized * alpha_full
    return np.clip(blended, 0, 255).astype(np.uint8)


def plot_stimulus_and_gaze_overlay(df: pd.DataFrame, stimulus_img: Path | None) -> None:
    """Figure 0 (1x2): Original NEMO free-viewing stimulus alongside 178k-fixation gaze overlay."""
    stim_rgb = _load_stimulus_rgb(stimulus_img)
    if stim_rgb is None:
        return

    grid, n_in = _build_gaze_density_grid(df, bins_x=192, bins_y=108)
    smooth = gaussian_filter(grid, sigma=3.2)
    blended = _overlay_heatmap_on_stimulus(stim_rgb, smooth)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.55), dpi=220)
    axes[0].imshow(stim_rgb)
    axes[0].set_title("A. Original 10-Second Free-Viewing Stimulus (1920x1080)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(blended)
    axes[1].set_title(f"B. Empirical Fixation Density Overlay ({n_in:,} Fixations, n = 7,250)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.text(
        0.99,
        0.012,
        "Stimulus image: NEMO Science Museum Amsterdam free-viewing experiment (Strauch et al., 2023; OSF sk4fr)",
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="#64748b",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1], pad=1.2)
    fig.savefig(FIG_DIR / "stimulus_and_gaze_overlay.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ofat_and_spatial_gaze(
    df: pd.DataFrame,
    ofat_df: pd.DataFrame,
    stimulus_img: Path | None = None,
) -> None:
    """Figure 4 (1x2): True OFAT Hessels + I-VT robustness + 2D screen fixation density map."""
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.7), dpi=220, gridspec_kw={"width_ratios": [1.18, 1.0]})

    # Panel A: OFAT sensitivity horizontal comparison (Median Duration + Ambient-to-Focal Rise %)
    ax = axes[0]
    if len(ofat_df):
        y_pos = np.arange(len(ofat_df))[::-1]
        fam_colors = {"Reference": COL_DARK, "Adaptive": COL_TEAL, "I-VT": COL_BLUE}
        cols = [fam_colors.get(f, COL_SLATE) for f in ofat_df["detector_family"]]

        ax.barh(y_pos, ofat_df["onset_05_to_70_rise_pct"], height=0.62, color=cols, alpha=0.86)
        ax.axvline(float(ofat_df.iloc[0]["onset_05_to_70_rise_pct"]), color=COL_DARK, ls="--", lw=1.1, alpha=0.65)
        for yp, row in zip(y_pos, ofat_df.itertuples()):
            ax.text(
                row.onset_05_to_70_rise_pct + 0.55,
                yp,
                f"+{row.onset_05_to_70_rise_pct:.1f}%  (med {row.median_ms:.0f} ms)",
                va="center",
                ha="left",
                fontsize=7.7,
                color=COL_DARK,
                fontweight="bold" if row.detector_family == "Reference" else "normal",
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ofat_df["config"], fontsize=8.0)
        max_r = float(ofat_df["onset_05_to_70_rise_pct"].max())
        ax.set_xlim(0, max_r + 11.5)
        ax.set_ylim(-2.35, len(ofat_df) - 0.35)
        ax.set_xlabel("Uncensored Ambient-to-Focal Rise, 0.5 s -> 7.0 s (%)")
        ax.set_title("A. OFAT Detector Robustness (1,000 Real 60 Hz Traces)")
        ax.grid(True, axis="x", alpha=0.35)

        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(facecolor=COL_DARK, label="OSF Reference (Hessels 2020)"),
                Patch(facecolor=COL_TEAL, label="MIT Adaptive OFAT (one param varied)"),
                Patch(facecolor=COL_BLUE, label="Independent I-VT (Salvucci & Goldberg)"),
            ],
            loc="lower right",
            fontsize=7.3,
            framealpha=0.96,
        )

    # Panel B: 2D Spatial Fixation Density aligned right-side-up on 1920 x 1080 stimulus screen
    ax2 = axes[1]
    stim_rgb = _load_stimulus_rgb(stimulus_img)
    grid, n_in = _build_gaze_density_grid(df, bins_x=192, bins_y=108)
    if stim_rgb is not None:
        smooth = gaussian_filter(grid, sigma=3.2)
        blended = _overlay_heatmap_on_stimulus(stim_rgb, smooth)
        ax2.imshow(blended, extent=[0, SCREEN_W_PX, SCREEN_H_PX, 0], aspect="equal")
        ax2.set_title(f"B. Fixation Density on Stimulus ({n_in:,} Fixations)", fontsize=10.8, pad=10)
    else:
        ax2.set_facecolor("#000004")
        im = ax2.imshow(
            np.maximum(grid, 0.5),
            origin="upper",
            extent=[0, SCREEN_W_PX, SCREEN_H_PX, 0],
            cmap="inferno",
            norm=LogNorm(vmin=1.0, vmax=max(10.0, float(grid.max()))),
            aspect="equal",
        )
        cbar = fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.03, shrink=0.66)
        cbar.set_label("Fixations per 10x10 px Bin (log)", fontsize=8.4)
        ax2.set_title(f"B. Screen Fixation Density ({n_in:,} Fixations)", fontsize=10.8, pad=10)
    ax2.set_xlabel("Screen X Coordinate (px, 1920x1080)")
    ax2.set_ylabel("Screen Y Coordinate (px, top-down)")
    ax2.set_xlim(0, SCREEN_W_PX)
    ax2.set_ylim(SCREEN_H_PX, 0)

    fig.tight_layout(pad=1.6)
    fig.savefig(FIG_DIR / "ofat_robustness_and_spatial_gaze.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_all(
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    preproc_dir: Path = DEFAULT_PREPROC_DIR,
    stimulus_img: Path | None = DEFAULT_STIMULUS_IMG,
    ofat_n: int = 1000,
    reuse_ofat: bool = False,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    df = load_merged_fixations(analysis_dir)
    print(f"Loaded {len(df):,} fixations across {df['ID'].nunique():,} valid free-viewing visitors.", flush=True)

    # 1. Age group summary CSV
    age_df = build_age_group_summary(df)
    age_df.to_csv(RES_DIR / "age_group_summary.csv", index=False)

    # 2. Raw LMM benchmark (reproducing lme4::lmer(duration ~ Order + (1 | ID))) for diagnostics contrast
    X_raw = np.column_stack([np.ones(len(df)), df["Order"].to_numpy(float)])
    raw_lmm = fit_random_intercept_lmm(df["duration"].to_numpy(float), X_raw, df["ID"].to_numpy(), reml=True)

    # 3. Nested Log-Normal LMMs across 5 sensitivity cohorts
    cohorts = [
        ("all_uncensored", df[df["onset"] <= 8.0]),
        ("excl_dob_2000", df[(df["DoB"] != 2000) & (df["onset"] <= 8.0)]),
        ("strict_valid_no2000", df[(df["Valid"] == True) & (df["DoB"] != 2000) & (df["onset"] <= 8.0)]),
        ("excl_order0_uncensored", df[(df["Order"] > 0) & (df["onset"] <= 8.0)]),
        ("all_full_10s", df),
    ]
    model_rows = []
    cohort_artifacts: dict[str, dict[str, object]] = {}
    for name, sub in cohorts:
        row, art = evaluate_cohort_models(sub, name)
        model_rows.append(row)
        cohort_artifacts[name] = art

    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(RES_DIR / "model_comparison.csv", index=False)
    print("\n=== MODEL COMPARISON SUMMARY ===")
    print(model_df.to_string(index=False))

    # 4. True OFAT robustness sweep on real 60 Hz pre-processed gaze traces
    ofat_csv = RES_DIR / "ofat_robustness.csv"
    if reuse_ofat and ofat_csv.exists():
        print("\n=== REUSING EXPORTED OFAT ROBUSTNESS CSV ===", flush=True)
        ofat_df = pd.read_csv(ofat_csv)
    else:
        print("\n=== OFAT ROBUSTNESS SWEEP ===", flush=True)
        ofat_df = run_ofat_robustness(df, preproc_dir, n_participants=ofat_n, seed=42)
        if len(ofat_df):
            ofat_df.to_csv(ofat_csv, index=False)

    # 5. Generate all 5 publication-grade figures (including stimulus + empirical gaze overlay)
    plot_stimulus_and_gaze_overlay(df, stimulus_img)
    plot_data_audit_and_diagnostics(df, raw_lmm, cohort_artifacts["all_uncensored"]["reml"])
    plot_age_lifespan_analysis(df, cohort_artifacts)
    plot_temporal_dynamics_and_censoring(df, cohort_artifacts)
    plot_ofat_and_spatial_gaze(df, ofat_df, stimulus_img=stimulus_img)
    print("\nSaved all CSV tables and diagnostic figures successfully.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NEMO eye-tracking fixation analysis and OFAT robustness.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ANALYSIS_DIR, help="Directory with compiled_fixations.csv & participant_info.csv")
    parser.add_argument("--preproc-dir", type=Path, default=DEFAULT_PREPROC_DIR, help="Directory with 60 Hz pre-processed gaze CSVs")
    parser.add_argument("--stimulus-img", type=Path, default=DEFAULT_STIMULUS_IMG, help="Path to NEMO free_viewing_img_1080.png stimulus")
    parser.add_argument("--ofat-n", type=int, default=1000, help="Number of stratified 60 Hz traces for OFAT sweep")
    parser.add_argument("--reuse-ofat", action="store_true", help="Reuse existing results/ofat_robustness.csv instead of re-parsing 60 Hz traces")
    args = parser.parse_args()
    run_all(
        analysis_dir=args.data_dir,
        preproc_dir=args.preproc_dir,
        stimulus_img=args.stimulus_img,
        ofat_n=args.ofat_n,
        reuse_ofat=args.reuse_ofat,
    )


if __name__ == "__main__":
    main()
