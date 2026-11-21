"""Unit tests for the exact REML LMM solver, spline LRTs, and velocity fixation detectors."""

from __future__ import annotations

import unittest
import numpy as np

from src.lmm import fit_random_intercept_lmm, likelihood_ratio_test, make_cubic_bspline
from src.fixation_detectors import detect_fixations_adaptive, detect_fixations_ivt, px_to_deg


class TestEyeTrackingPipeline(unittest.TestCase):
    def test_reml_lmm_recovers_known_fixed_and_random_parameters(self) -> None:
        rng = np.random.default_rng(123)
        n_groups = 120
        n_per = 25
        groups = np.repeat(np.arange(n_groups), n_per)
        u_true = rng.normal(0.0, 0.35, size=n_groups)
        x = rng.uniform(0.0, 8.0, size=n_groups * n_per)
        eps = rng.normal(0.0, 0.50, size=n_groups * n_per)
        y = -1.6 + 0.04 * x + u_true[groups] + eps

        X = np.column_stack([np.ones_like(x), x])
        fit = fit_random_intercept_lmm(y, X, groups, reml=True)

        self.assertAlmostEqual(float(fit.beta[0]), -1.6, delta=0.08)
        self.assertAlmostEqual(float(fit.beta[1]), 0.04, delta=0.015)
        self.assertAlmostEqual(fit.sigma_u, 0.35, delta=0.06)
        self.assertAlmostEqual(fit.sigma_e, 0.50, delta=0.04)
        self.assertGreater(fit.r2_conditional, fit.r2_marginal)

    def test_likelihood_ratio_test_detects_nonlinear_spline(self) -> None:
        rng = np.random.default_rng(77)
        n_groups = 80
        n_per = 20
        groups = np.repeat(np.arange(n_groups), n_per)
        u = rng.normal(0.0, 0.15, size=n_groups)
        t = rng.uniform(0.0, 8.0, size=n_groups * n_per)
        # Clear nonlinear saturation curve
        y = -1.65 + 0.25 * (1.0 - np.exp(-t / 2.2)) + u[groups] + rng.normal(0.0, 0.25, size=len(t))

        B_spl, _ = make_cubic_bspline(t, df=5)
        m_lin = fit_random_intercept_lmm(y, np.column_stack([np.ones_like(t), t]), groups, reml=False)
        m_spl = fit_random_intercept_lmm(y, np.column_stack([np.ones_like(t), B_spl]), groups, reml=False)
        stat, df, pval = likelihood_ratio_test(m_spl, m_lin)

        self.assertEqual(df, 4)
        self.assertGreater(stat, 15.0)
        self.assertLess(pval, 1e-3)

    def test_lrt_rejects_reml_inputs(self) -> None:
        rng = np.random.default_rng(5)
        g = np.repeat(np.arange(15), 10)
        y = rng.normal(size=150)
        m0 = fit_random_intercept_lmm(y, np.ones((150, 1)), g, reml=True)
        m1 = fit_random_intercept_lmm(y, np.column_stack([np.ones(150), rng.normal(size=150)]), g, reml=True)
        with self.assertRaises(ValueError):
            likelihood_ratio_test(m1, m0)

    def test_adaptive_and_ivt_detectors_recover_synthetic_fixations(self) -> None:
        # Construct a clean 60 Hz gaze trace with 5 distinct 0.30 s fixations separated by 6-deg saccades
        rng = np.random.default_rng(99)
        fs = 60.0
        dt = 1.0 / fs
        targets = [(400.0, 540.0), (750.0, 540.0), (1100.0, 350.0), (1400.0, 700.0), (960.0, 540.0)]
        xs, ys = [], []
        for idx, (tx, ty) in enumerate(targets):
            # 18 samples = 300 ms steady gaze with small sensor jitter
            xs.append(tx + rng.normal(0.0, 1.2, size=18))
            ys.append(ty + rng.normal(0.0, 1.2, size=18))
            if idx + 1 < len(targets):
                nx, ny = targets[idx + 1]
                # 3-sample saccade ramp (50 ms)
                xs.append(np.linspace(tx, nx, 5)[1:-1])
                ys.append(np.linspace(ty, ny, 5)[1:-1])
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        t = np.arange(len(x)) * dt

        fx_ad = detect_fixations_adaptive(x, y, t, lam=2.5, min_fix_s=0.06, min_amp_deg=1.0)
        fx_iv = detect_fixations_ivt(x, y, t, vel_thr_deg_s=30.0, min_fix_s=0.06, min_amp_deg=1.0)

        self.assertEqual(len(fx_ad), 5)
        self.assertEqual(len(fx_iv), 5)
        self.assertTrue(np.all(fx_ad["duration"].between(0.20, 0.36)))
        self.assertTrue(np.all(fx_iv["duration"].between(0.20, 0.36)))
        self.assertGreater(float(px_to_deg(350.0)), 5.0)


if __name__ == "__main__":
    unittest.main()
