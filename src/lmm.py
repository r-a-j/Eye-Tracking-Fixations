"""Exact REML/ML random-intercept Linear Mixed-Effects Model solver and spline utilities.

Solves Gaussian linear mixed-effects models of the form:
    y_ij = X_ij beta + u_i + eps_ij,   u_i ~ N(0, sigma_u^2),   eps_ij ~ N(0, sigma_e^2)
using the exact block-diagonal Sherman-Morrison-Woodbury identity for V^{-1} and log|V|,
producing exact REML and ML estimates numerically matching lme4::lmer(..., (1 | group)).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar
from scipy import stats


@dataclass
class LMMResult:
    beta: np.ndarray
    se: np.ndarray
    t_values: np.ndarray
    p_values_wald: np.ndarray
    cov_beta: np.ndarray
    sigma_u: float
    sigma_e: float
    icc: float
    r2_marginal: float
    r2_conditional: float
    loglik: float
    aic: float
    bic: float
    n_obs: int
    n_groups: int
    n_params: int
    reml: bool
    u_hat: np.ndarray
    fitted_fixed: np.ndarray
    fitted_cond: np.ndarray
    residuals: np.ndarray
    resid_skew: float
    resid_kurtosis: float
    max_abs_std_resid: float


def fit_random_intercept_lmm(
    y: np.ndarray,
    X: np.ndarray,
    groups: np.ndarray,
    *,
    reml: bool = True,
) -> LMMResult:
    """Fit exact REML or ML random-intercept LMM via profiled variance ratio lambda = sigma_u^2 / sigma_e^2."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape

    _, g_idx, g_counts = np.unique(groups, return_inverse=True, return_counts=True)
    n_groups = len(g_counts)
    n_g = g_counts.astype(float)

    # Sort by group once so group sums reduce in O(n)
    order = np.argsort(g_idx, kind="mergesort")
    y_s = y[order]
    X_s = X[order]
    starts = np.concatenate([[0], np.cumsum(g_counts)[:-1]])
    y_sum = np.add.reduceat(y_s, starts)
    X_sum = np.add.reduceat(X_s, starts, axis=0)

    XtX_full = X_s.T @ X_s
    Xty_full = X_s.T @ y_s
    yty_full = float(y_s @ y_s)

    def _neg_loglik(log_lam: float) -> float:
        lam = np.exp(log_lam)
        w = lam / (1.0 + n_g * lam)
        XtVinvX = XtX_full - (X_sum.T * w) @ X_sum
        XtVinvy = Xty_full - (X_sum.T * w) @ y_sum
        beta = np.linalg.solve(XtVinvX, XtVinvy)

        r_sum = y_sum - X_sum @ beta
        r_full_sq = yty_full - 2.0 * float(beta @ Xty_full) + float(beta @ XtX_full @ beta)
        quad = max(r_full_sq - float(np.sum(w * (r_sum**2))), 1e-18)
        logdet_V = float(np.sum(np.log(1.0 + n_g * lam)))

        if reml:
            _, logdet_X = np.linalg.slogdet(XtVinvX)
            df_eff = n - p
            s2 = quad / df_eff
            return 0.5 * (logdet_V + logdet_X + df_eff * (1.0 + np.log(2.0 * np.pi * s2)))
        s2 = quad / n
        return 0.5 * (logdet_V + n * (1.0 + np.log(2.0 * np.pi * s2)))

    opt = minimize_scalar(_neg_loglik, bounds=(-14.0, 6.0), method="bounded")
    lam = float(np.exp(opt.x))
    w = lam / (1.0 + n_g * lam)

    XtVinvX = XtX_full - (X_sum.T * w) @ X_sum
    XtVinvy = Xty_full - (X_sum.T * w) @ y_sum
    beta = np.linalg.solve(XtVinvX, XtVinvy)

    r_sum = y_sum - X_sum @ beta
    r_full_sq = yty_full - 2.0 * float(beta @ Xty_full) + float(beta @ XtX_full @ beta)
    quad = max(r_full_sq - float(np.sum(w * (r_sum**2))), 1e-18)

    df_eff = (n - p) if reml else n
    sigma_e2 = quad / df_eff
    sigma_u2 = lam * sigma_e2
    sigma_u = float(np.sqrt(sigma_u2))
    sigma_e = float(np.sqrt(sigma_e2))

    cov_beta = sigma_e2 * np.linalg.inv(XtVinvX)
    se = np.sqrt(np.diag(cov_beta))
    t_vals = beta / se
    p_wald = 2.0 * stats.norm.sf(np.abs(t_vals))

    # Empirical Bayes random intercepts (BLUPs)
    u_hat = (lam / (1.0 + n_g * lam)) * r_sum
    fitted_fixed = X @ beta
    fitted_cond = fitted_fixed + u_hat[g_idx]
    residuals = y - fitted_cond

    # Nakagawa & Schielzeth (2013) marginal and conditional R^2
    var_fixed = float(np.var(fitted_fixed))
    var_total = var_fixed + sigma_u2 + sigma_e2
    r2_marg = var_fixed / var_total if var_total > 0 else 0.0
    r2_cond = (var_fixed + sigma_u2) / var_total if var_total > 0 else 0.0

    loglik = -float(opt.fun)
    k_total = p + 2  # p fixed effects + sigma_u^2 + sigma_e^2
    aic = 2.0 * k_total - 2.0 * loglik
    bic = k_total * np.log(n) - 2.0 * loglik

    return LMMResult(
        beta=beta,
        se=se,
        t_values=t_vals,
        p_values_wald=p_wald,
        cov_beta=cov_beta,
        sigma_u=sigma_u,
        sigma_e=sigma_e,
        icc=float(sigma_u2 / (sigma_u2 + sigma_e2)),
        r2_marginal=float(r2_marg),
        r2_conditional=float(r2_cond),
        loglik=loglik,
        aic=float(aic),
        bic=float(bic),
        n_obs=n,
        n_groups=n_groups,
        n_params=k_total,
        reml=reml,
        u_hat=u_hat,
        fitted_fixed=fitted_fixed,
        fitted_cond=fitted_cond,
        residuals=residuals,
        resid_skew=float(stats.skew(residuals)),
        resid_kurtosis=float(stats.kurtosis(residuals)),
        max_abs_std_resid=float(np.max(np.abs(residuals)) / sigma_e),
    )


def likelihood_ratio_test(full_ml: LMMResult, reduced_ml: LMMResult) -> tuple[float, int, float]:
    """Exact chi-square Likelihood Ratio Test comparing two nested ML LMM fits."""
    if full_ml.reml or reduced_ml.reml:
        raise ValueError("Likelihood ratio tests for fixed effects require ML (reml=False) fits.")
    df = full_ml.n_params - reduced_ml.n_params
    if df <= 0:
        raise ValueError("Full model must have strictly more parameters than reduced model.")
    stat = max(0.0, 2.0 * (full_ml.loglik - reduced_ml.loglik))
    pval = float(stats.chi2.sf(stat, df))
    return stat, df, pval


def make_cubic_bspline(
    x: np.ndarray,
    df: int = 5,
    *,
    knots: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct cubic B-spline basis (dropping first column for use with an intercept)."""
    x = np.asarray(x, dtype=float)
    if knots is None:
        lo, hi = float(np.min(x)), float(np.max(x))
        n_interior = max(0, df - 3)
        if n_interior > 0:
            qs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
            interior = np.quantile(x, qs)
        else:
            interior = np.array([], dtype=float)
        knots = np.concatenate([[lo] * 4, interior, [hi] * 4])

    clipped = np.clip(x, knots[0], knots[-1])
    basis = BSpline.design_matrix(clipped, knots, 3).toarray()
    return basis[:, 1:], knots
