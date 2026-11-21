"""Generate large-font, single-panel standalone figures for the NEMO eye-tracking study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.ndimage import gaussian_filter

from src.lmm import LMMResult, fit_random_intercept_lmm, make_cubic_bspline
from src.run_analysis import (
    COL_AMBER,
    COL_BLUE,
    COL_DARK,
    COL_PURPLE,
    COL_RED,
    COL_SLATE,
    COL_TEAL,
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_STIMULUS_IMG,
    _build_gaze_density_grid,
    _load_stimulus_rgb,
    _overlay_heatmap_on_stimulus,
    load_merged_fixations,
)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 1.0,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.titlesize": 13.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.2,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
            "legend.fontsize": 9.8,
            "grid.color": "#e2e8f0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.75,
        }
    )


def _save(fig: plt.Figure, png_path: Path, webp_dir: Path | None, webp_name: str) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    if webp_dir is not None:
        webp_dir.mkdir(parents=True, exist_ok=True)
        Image.open(png_path).convert("RGB").save(
            webp_dir / webp_name, format="WEBP", quality=92, method=6
        )


def _fit_cohort_reml(sub: pd.DataFrame) -> dict[str, object]:
    B_on, on_knots = make_cubic_bspline(sub["onset"].to_numpy(float), df=5)
    B_ag, age_knots = make_cubic_bspline(sub["age"].to_numpy(float), df=5)
    X = np.column_stack([np.ones(len(sub)), B_on, B_ag])
    reml = fit_random_intercept_lmm(
        sub["log_dur"].to_numpy(float), X, sub["ID"].to_numpy(), reml=True
    )
    return {"reml": reml, "on_knots": on_knots, "age_knots": age_knots}


def build_eye_panels(
    analysis_dir: Path,
    stimulus_img: Path,
    results_dir: Path,
    png_dir: Path,
    webp_dir: Path | None = None,
) -> None:
    _apply_style()
    df = load_merged_fixations(analysis_dir)
    ofat_df = pd.read_csv(results_dir / "ofat_robustness.csv")

    # 1 & 2. Standalone Original Stimulus + Pixel-Exact Gaze Density Overlay
    stim_rgb = _load_stimulus_rgb(stimulus_img)
    if stim_rgb is not None:
        grid, n_in = _build_gaze_density_grid(df, bins_x=192, bins_y=108)
        smooth = gaussian_filter(grid, sigma=3.2)
        blended = _overlay_heatmap_on_stimulus(stim_rgb, smooth)

        fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=220)
        ax.imshow(stim_rgb)
        ax.set_title("Original 10-Second Free-Viewing Stimulus (1920x1080)", loc="left", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        _save(fig, png_dir / "stimulus_original.png", webp_dir, "stimulus-original.webp")

        fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=220)
        ax.imshow(blended)
        ax.set_title(f"Empirical Gaze Density Overlay ({n_in:,} Fixations, n = 7,250)", loc="left", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        _save(fig, png_dir / "stimulus_gaze_overlay.png", webp_dir, "stimulus-gaze-overlay.webp")

    # 3. Kiosk DoB = 2000 Default Spike Audit
    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=220)
    p_ages = df.groupby("ID")["age"].first().to_numpy(int)
    age_vals = np.arange(3, 86)
    counts = np.array([int(np.sum(p_ages == a)) for a in age_vals])
    cols = [COL_RED if a == 23 else COL_BLUE for a in age_vals]
    ax.bar(age_vals, counts, width=0.84, color=cols, alpha=0.88, edgecolor="none")
    n_23 = int(np.sum(p_ages == 23))
    ax.annotate(
        f"DoB = 2000 default spike\nAge 23: n = {n_23:,} ({100.0 * n_23 / len(p_ages):.1f}%)\n~5x adjacent cohorts",
        xy=(23, n_23),
        xytext=(36, n_23 * 0.78),
        arrowprops=dict(arrowstyle="->", color=COL_RED, lw=1.6),
        fontsize=10.2,
        fontweight="bold",
        color=COL_RED,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#fecaca", alpha=0.96),
    )
    ax.set_title("Participant Age Audit: Kiosk DoB = 2000 Spike\n(n = 7,250 Valid Visitors, NEMO Science Museum)", loc="left", pad=10)
    ax.set_xlabel("Reported Age (years)")
    ax.set_ylabel("Participant Count")
    ax.set_xlim(1, 86)
    ax.grid(True, axis="y", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, png_dir / "dob_2000_spike.png", webp_dir, "dob-2000-spike.webp")

    # 4. Raw Gaussian vs Log-Normal LMM Residual Q-Q Overlay (single super-clear chart!)
    X_ord = np.column_stack([np.ones(len(df)), df["Order"].to_numpy(float)])
    raw_lmm = fit_random_intercept_lmm(df["duration"].to_numpy(float), X_ord, df["ID"].to_numpy(), reml=True)
    log_lmm = fit_random_intercept_lmm(df["log_dur"].to_numpy(float), X_ord, df["ID"].to_numpy(), reml=True)

    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=220)
    q_idx = np.linspace(0.0005, 0.9995, 2500)
    theo_q = stats.norm.ppf(q_idx)
    emp_raw = np.quantile(raw_lmm.residuals / raw_lmm.sigma_e, q_idx)
    emp_log = np.quantile(log_lmm.residuals / log_lmm.sigma_e, q_idx)
    ax.plot([-3.7, 3.7], [-3.7, 3.7], "--", color=COL_DARK, lw=1.6, label="Ideal Normal Reference (y = x)")
    ax.scatter(theo_q, emp_raw, s=16, color=COL_RED, alpha=0.72, edgecolors="none",
               label=f"Raw Duration LMM (Skew {raw_lmm.resid_skew:.2f}, Kurt {raw_lmm.resid_kurtosis:.1f}, Max {raw_lmm.max_abs_std_resid:.1f} SD)")
    ax.scatter(theo_q, emp_log, s=16, color=COL_TEAL, alpha=0.82, edgecolors="none",
               label=f"Log-Normal LMM (Skew {log_lmm.resid_skew:.2f}, Kurt {log_lmm.resid_kurtosis:.2f}, Max {log_lmm.max_abs_std_resid:.1f} SD)")
    ax.set_title("Standardized Residual Q-Q: Raw vs. Log-Normal LMM\n(n = 183,627 Fixations Across 7,250 Visitors)", loc="left", pad=10)
    ax.set_xlabel("Theoretical Standard Normal Quantiles")
    ax.set_ylabel("Standardized Residual Quantiles (z)")
    ax.set_xlim(-3.7, 3.7)
    ax.set_ylim(-4.2, 11.5)
    ax.grid(True, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.1)
    _save(fig, png_dir / "lmm_residual_qq.png", webp_dir, "lmm-residual-qq.webp")

    # Fit the 4 cohorts needed for age & temporal standalone panels
    art_all = _fit_cohort_reml(df[df["onset"] <= 8.0])
    art_no2k = _fit_cohort_reml(df[(df["onset"] <= 8.0) & (df["DoB"] != 2000)])
    art_strict = _fit_cohort_reml(df[(df["onset"] <= 8.0) & (df["DoB"] != 2000) & (df["Valid"] == True)])
    art_full10 = _fit_cohort_reml(df)

    # 5. Continuous Lifespan Age Splines Across 3 Cohorts
    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=220)
    age_grid = np.linspace(5.0, 75.0, 141)
    sub_unc = df[df["onset"] <= 8.0]
    ps = sub_unc.groupby("ID").agg(log_mean=("log_dur", "mean"), age=("age", "first")).reset_index()
    ps["abin"] = pd.cut(ps["age"], bins=[3, 8, 12, 16, 20, 25, 30, 36, 42, 48, 55, 63, 80])
    bx, by, bse = [], [], []
    for _, grp in ps.groupby("abin", observed=False):
        if len(grp) >= 20:
            mu = float(grp["log_mean"].mean())
            se = float(grp["log_mean"].std(ddof=1) / np.sqrt(len(grp)))
            bx.append(float(grp["age"].mean()))
            by.append(np.exp(mu) * 1000.0)
            bse.append(1.96 * se * np.exp(mu) * 1000.0)
    ax.errorbar(bx, by, yerr=bse, fmt="o", color="#64748b", ecolor="#94a3b8", elinewidth=1.4, capsize=3, markersize=5.5,
                alpha=0.9, label="Binned viewer geom. mean (± 95% CI)", zorder=3)
    specs = [
        (art_all, "All valid free-viewing (n=7,250; p=0.123)", COL_BLUE, "-", True),
        (art_no2k, "Excl. DoB=2000 spike (n=5,718; p=0.085)", COL_TEAL, "--", False),
        (art_strict, "Strict Valid & DoB!=2000 (n=4,546; p=0.024)", COL_AMBER, "-.", False),
    ]
    for art, lbl, col, ls, show_ci in specs:
        m: LMMResult = art["reml"]
        B_on_r, _ = make_cubic_bspline(np.full_like(age_grid, 4.0), df=5, knots=art["on_knots"])
        B_ag_g, _ = make_cubic_bspline(age_grid, df=5, knots=art["age_knots"])
        Xg = np.column_stack([np.ones(len(age_grid)), B_on_r, B_ag_g])
        lp = Xg @ m.beta
        se_lp = np.sqrt(np.einsum("ij,jk,ik->i", Xg, m.cov_beta, Xg))
        ax.plot(age_grid, np.exp(lp) * 1000.0, color=col, ls=ls, lw=2.5, label=lbl, zorder=5)
        if show_ci:
            ax.fill_between(age_grid, np.exp(lp - 1.96 * se_lp) * 1000.0, np.exp(lp + 1.96 * se_lp) * 1000.0, color=col, alpha=0.15, zorder=2)
    ax.set_title("Continuous Lifespan Age Splines Across 3 Cohorts\n(Log-Normal LMM at onset = 4 s; No Late-Life U-Curve)", loc="left", pad=10)
    ax.set_xlabel("Participant Age (years)")
    ax.set_ylabel("Predicted Median Fixation Duration (ms)")
    ax.set_xlim(4, 76)
    ax.set_ylim(185, 245)
    ax.grid(True, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.1)
    _save(fig, png_dir / "age_lifespan_splines.png", webp_dir, "age-lifespan-splines.webp")

    # 6. Between-Viewer Baseline Variability (BLUPs)
    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=220)
    m_main: LMMResult = art_all["reml"]
    B_on_r, _ = make_cubic_bspline(np.array([4.0]), df=5, knots=art_all["on_knots"])
    B_ag_r, _ = make_cubic_bspline(np.array([25.0]), df=5, knots=art_all["age_knots"])
    base_lp = float((np.column_stack([np.ones(1), B_on_r, B_ag_r]) @ m_main.beta).item())
    subj_ms = np.exp(base_lp + m_main.u_hat) * 1000.0
    p10, p50, p90 = np.percentile(subj_ms, [10, 50, 90])
    ax.hist(subj_ms, bins=48, color=COL_PURPLE, alpha=0.74, edgecolor="white", linewidth=0.45, density=True)
    ax.axvline(p10, color=COL_DARK, ls=":", lw=1.8, label=f"10th pct viewer: {p10:.0f} ms")
    ax.axvline(p50, color=COL_DARK, ls="-", lw=2.0, label=f"Median viewer: {p50:.0f} ms")
    ax.axvline(p90, color=COL_DARK, ls="--", lw=1.8, label=f"90th pct viewer: {p90:.0f} ms")
    ax.axvspan(208.3, 220.4, color=COL_RED, alpha=0.25, label="Entire 70-yr age range (208-220 ms)")
    ax.text(0.97, 0.50, f"sigma_u = {m_main.sigma_u:.3f} (log scale)\nICC = {100.0 * m_main.icc:.1f}%\n10th-90th spread: {p90 - p10:.0f} ms\nvs. ~12 ms across ages 6-75",
            transform=ax.transAxes, ha="right", va="center", fontsize=9.8,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#cbd5e1", alpha=0.96))
    ax.set_title("Between-Viewer Baseline Variability vs. Age Shift\n(Random-Intercept BLUPs Across 7,250 Visitors)", loc="left", pad=10)
    ax.set_xlabel("Participant Baseline Median Fixation Duration (ms, BLUPs)")
    ax.set_ylabel("Density Across 7,250 Visitors")
    ax.set_xlim(140, 320)
    ax.grid(True, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.2)
    _save(fig, png_dir / "viewer_baseline_blups.png", webp_dir, "viewer-baseline-blups.webp")

    # 7. Continuous Trial Onset & 10-Second Right-Censoring
    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=220)
    edges = np.linspace(0.0, 10.0, 21)
    mids = 0.5 * (edges[:-1] + edges[1:])
    df_tmp = df.copy()
    df_tmp["obin"] = pd.cut(df_tmp["onset"], bins=edges)
    grp = df_tmp.groupby("obin", observed=False).agg(geom_ms=("log_dur", lambda s: np.exp(s.mean()) * 1000.0))
    ax.axvspan(8.0, 10.0, color="#fee2e2", alpha=0.58, zorder=0, label="Right-censoring zone (offset <= 10 s)")
    ax.axvline(8.0, color=COL_RED, ls="--", lw=1.4, alpha=0.85)
    ax.scatter(mids[mids <= 8.0], grp["geom_ms"].to_numpy()[mids <= 8.0], s=44, color=COL_TEAL, zorder=4, label="Empirical bin geom. mean (uncensored)")
    ax.scatter(mids[mids > 8.0], grp["geom_ms"].to_numpy()[mids > 8.0], s=48, color=COL_RED, marker="x", lw=2.1, zorder=4, label="Empirical bin geom. mean (censored tail)")

    t_u = np.linspace(0.05, 8.0, 160)
    B_on_u, _ = make_cubic_bspline(t_u, df=5, knots=art_all["on_knots"])
    B_ag_u, _ = make_cubic_bspline(np.full_like(t_u, 25.0), df=5, knots=art_all["age_knots"])
    Xu = np.column_stack([np.ones(len(t_u)), B_on_u, B_ag_u])
    lp_u = Xu @ art_all["reml"].beta
    se_u = np.sqrt(np.einsum("ij,jk,ik->i", Xu, art_all["reml"].cov_beta, Xu))
    ax.plot(t_u, np.exp(lp_u) * 1000.0, color=COL_TEAL, lw=2.8, label="Log-Normal LMM spline (0-8 s uncensored)", zorder=5)
    ax.fill_between(t_u, np.exp(lp_u - 1.96 * se_u) * 1000.0, np.exp(lp_u + 1.96 * se_u) * 1000.0, color=COL_TEAL, alpha=0.18, zorder=3)

    t_f = np.linspace(0.05, 9.75, 195)
    B_on_f, _ = make_cubic_bspline(t_f, df=5, knots=art_full10["on_knots"])
    B_ag_f, _ = make_cubic_bspline(np.full_like(t_f, 25.0), df=5, knots=art_full10["age_knots"])
    Xf = np.column_stack([np.ones(len(t_f)), B_on_f, B_ag_f])
    ax.plot(t_f, np.exp(Xf @ art_full10["reml"].beta) * 1000.0, color=COL_RED, ls=":", lw=2.2, label="Naive 0-10 s spline (cutoff bias)", zorder=4)

    B_ev, _ = make_cubic_bspline(np.array([0.5, 7.0]), df=5, knots=art_all["on_knots"])
    B_ag_ev, _ = make_cubic_bspline(np.array([25.0, 25.0]), df=5, knots=art_all["age_knots"])
    pred_ev = np.exp(np.column_stack([np.ones(2), B_ev, B_ag_ev]) @ art_all["reml"].beta) * 1000.0
    rise_pct = 100.0 * (pred_ev[1] / pred_ev[0] - 1.0)
    ax.annotate(f"Ambient-to-focal rise\n(+{rise_pct:.1f}% from 0.5 s to 7.0 s)\nthen plateau",
                xy=(6.8, float(pred_ev[1])), xytext=(2.6, 239.0),
                arrowprops=dict(arrowstyle="->", color=COL_TEAL, lw=1.4), fontsize=9.6, fontweight="bold", color="#065f46",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#a7f3d0", alpha=0.96))
    ax.annotate("96.9% final\nfixations", xy=(9.65, 145.0), xytext=(7.2, 154.0),
                arrowprops=dict(arrowstyle="->", color=COL_RED, lw=1.3), fontsize=9.2, fontweight="bold", color=COL_RED)
    ax.set_title("Elapsed Trial Time (0-10 s) & Right-Censoring\n(Ambient-to-Focal Progression; LRT p = 9.2e-248)", loc="left", pad=10)
    ax.set_xlabel("Fixation Onset Time (seconds)")
    ax.set_ylabel("Median / Geometric Mean Duration (ms)")
    ax.set_xlim(0, 10.0)
    ax.set_ylim(130, 260)
    ax.grid(True, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.6)
    _save(fig, png_dir / "trial_onset_and_censoring.png", webp_dir, "trial-onset-and-censoring.webp")

    # 8. Fixation-Order Survivorship Bias
    fig, ax = plt.subplots(figsize=(8.4, 5.8), dpi=220)
    ord_grp = df.groupby("Order").agg(mean_ms=("duration_ms", "mean"), geom_ms=("log_dur", lambda s: np.exp(s.mean()) * 1000.0), n_active=("ID", "nunique")).reset_index()
    ord_grp = ord_grp[ord_grp["Order"] <= 36]
    ord_grp["pct_active"] = 100.0 * ord_grp["n_active"] / df["ID"].nunique()
    l1 = ax.plot(ord_grp["Order"], ord_grp["mean_ms"], "-o", ms=4.5, color=COL_BLUE, lw=2.3, label="Mean fixation duration (ms)")
    l2 = ax.plot(ord_grp["Order"], ord_grp["geom_ms"], "--", color=COL_TEAL, lw=2.1, label="Geom. mean duration (ms)")
    ax.set_xlabel("Fixation Order Index (0 = Trial Start)")
    ax.set_ylabel("Fixation Duration (ms)", color=COL_DARK)
    ax.set_xlim(0, 36)
    ax.set_ylim(115, 295)
    ax.grid(True, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax_r = ax.twinx()
    l3 = ax_r.fill_between(ord_grp["Order"], 0, ord_grp["pct_active"], color="#94a3b8", alpha=0.22, label="% of 7,250 viewers reaching Order k")
    ax_r.plot(ord_grp["Order"], ord_grp["pct_active"], color="#64748b", ls=":", lw=1.7)
    ax_r.set_ylabel("Active Participants Remaining (%)", color="#475569")
    ax_r.set_ylim(0, 108)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_color("#94a3b8")
    ax.annotate("Selection bias past Order 22:\nonly fast-saccading viewers\nfit 25-36 fixations in 10 s",
                xy=(29, 224), xytext=(9.5, 138),
                arrowprops=dict(arrowstyle="->", color=COL_BLUE, lw=1.4), fontsize=9.6, fontweight="bold", color="#1e3a8a",
                bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#bfdbfe", alpha=0.96))
    lines = l1 + l2 + [l3]
    ax.legend(lines, [x.get_label() for x in lines], loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.0)
    ax.set_title("Why Fixation Order Misleads in Fixed-Length Trials\n(Survivorship Bias as Active Viewer Share Collapses)", loc="left", pad=10)
    _save(fig, png_dir / "order_survivorship_bias.png", webp_dir, "order-survivorship-bias.webp")

    # 9. Standalone 12-Config OFAT + I-VT Detector Robustness (wide full-width hero!)
    fig, ax = plt.subplots(figsize=(10.6, 5.6), dpi=220)
    y_pos = np.arange(len(ofat_df))[::-1]
    fam_colors = {"Reference": COL_DARK, "Adaptive": COL_TEAL, "I-VT": COL_BLUE}
    cols = [fam_colors.get(f, COL_SLATE) for f in ofat_df["detector_family"]]
    ax.barh(y_pos, ofat_df["onset_05_to_70_rise_pct"], height=0.62, color=cols, alpha=0.88)
    ax.axvline(float(ofat_df.iloc[0]["onset_05_to_70_rise_pct"]), color=COL_DARK, ls="--", lw=1.3, alpha=0.7)
    for yp, row in zip(y_pos, ofat_df.itertuples()):
        ax.text(row.onset_05_to_70_rise_pct + 0.35, yp, f"+{row.onset_05_to_70_rise_pct:.1f}%   (median {row.median_ms:.0f} ms, age {row.age_pct_per_decade:+.1f}%/dec)",
                va="center", ha="left", fontsize=9.4, color=COL_DARK, fontweight="bold" if row.detector_family == "Reference" else "normal")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ofat_df["config"], fontsize=9.8, fontweight="bold")
    ax.set_xlim(0, float(ofat_df["onset_05_to_70_rise_pct"].max()) + 12.0)
    ax.set_xlabel("Uncensored Ambient-to-Focal Rise, 0.5 s -> 7.0 s (%)")
    ax.set_title("12-Configuration OFAT & I-VT Detector Robustness (1,000 Real 60 Hz Gaze Traces)", loc="left", pad=10)
    ax.grid(True, axis="x", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=COL_DARK, label="OSF Reference (Hessels 2020)"),
            Patch(facecolor=COL_TEAL, label="MIT Adaptive OFAT (one param varied)"),
            Patch(facecolor=COL_BLUE, label="Independent I-VT (Salvucci & Goldberg)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.48, -0.14),
        ncol=3,
        frameon=False,
        fontsize=9.6,
    )
    _save(fig, png_dir / "ofat_detector_robustness.png", webp_dir, "ofat-detector-robustness.webp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--stimulus-img", type=Path, default=DEFAULT_STIMULUS_IMG)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--png-dir", type=Path, default=Path("figures/panels"))
    parser.add_argument("--webp-dir", type=Path, default=None)
    args = parser.parse_args()
    build_eye_panels(args.analysis_dir, args.stimulus_img, args.results_dir, args.png_dir, args.webp_dir)
    print("Built all 9 standalone Eye-Tracking panels cleanly.")
