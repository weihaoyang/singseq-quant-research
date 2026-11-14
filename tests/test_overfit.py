"""Backtest-overfitting gates: DSR, PBO(CSCV), multiple-testing haircut."""

import unittest

import numpy as np

from evaluation import overfit


class DeflatedSharpeTests(unittest.TestCase):
    def test_strong_strategy_high_dsr_and_monotone_in_trials(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0010, 0.01, 1000)          # SR ~ 1.6 annualised
        few = overfit.deflated_sharpe_ratio(returns, n_trials=1)
        many = overfit.deflated_sharpe_ratio(returns, n_trials=1000)
        self.assertGreater(few["dsr"], 0.9)
        self.assertGreater(few["dsr"], many["dsr"])       # more trials -> more deflation
        self.assertGreater(many["expected_max_sr"], few["expected_max_sr"])

    def test_noise_strategy_is_deflated(self):
        rng = np.random.default_rng(2)
        returns = rng.normal(0.0, 0.01, 500)
        result = overfit.deflated_sharpe_ratio(returns, n_trials=50)
        self.assertLess(result["dsr"], 0.5)

    def test_var_sr_from_trial_returns(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0.001, 0.01, 400)
        trials = [rng.normal(0.0, 0.01, 400) for _ in range(20)]
        result = overfit.deflated_sharpe_ratio(returns, n_trials=len(trials), trial_returns=trials)
        self.assertEqual(result["var_sr_source"], "trial_returns")
        self.assertGreaterEqual(result["dsr"], 0.0)

    def test_trial_returns_must_match_declared_aligned_family(self):
        returns = np.full(20, 0.001)
        result = overfit.deflated_sharpe_ratio(
            returns, n_trials=2, trial_returns=[returns])
        self.assertFalse(result["valid"])
        self.assertIn("exactly n_trials", result["reason"])

    def test_non_finite_observation_does_not_get_dropped(self):
        rng = np.random.default_rng(33)
        returns = rng.normal(0.001, 0.01, 100)
        returns[4] = np.nan
        result = overfit.deflated_sharpe_ratio(returns, n_trials=2)
        self.assertFalse(result["valid"])
        self.assertEqual(result["dsr"], 0.0)
        self.assertIn("finite", result["reason"])

    def test_invalid_trial_count_fails_closed(self):
        result = overfit.deflated_sharpe_ratio([0.01, 0.02, 0.0], n_trials=0)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["n_trials"])


class PboTests(unittest.TestCase):
    def test_dominant_strategy_has_low_pbo(self):
        rng = np.random.default_rng(4)
        t, n = 640, 6
        matrix = rng.normal(0.0, 0.01, (t, n))
        matrix[:, 0] += 0.002                              # strategy 0 dominates
        result = overfit.pbo_cscv(matrix, n_splits=16)
        self.assertIsNotNone(result["pbo"])
        self.assertLess(result["pbo"], 0.2)

    def test_pure_noise_has_higher_pbo(self):
        rng = np.random.default_rng(5)
        matrix = rng.normal(0.0, 0.01, (640, 8))
        result = overfit.pbo_cscv(matrix, n_splits=16)
        self.assertGreater(result["pbo"], 0.3)

    def test_insufficient_data_returns_none(self):
        result = overfit.pbo_cscv(np.zeros((4, 3)), n_splits=16)
        self.assertIsNone(result["pbo"])

    def test_non_finite_candidate_matrix_fails_closed(self):
        matrix = np.ones((64, 2), dtype=float)
        matrix[7, 1] = np.nan
        result = overfit.pbo_cscv(matrix, n_splits=8)
        self.assertIsNone(result["pbo"])
        self.assertIn("non-finite", result["note"])

    def test_invalid_split_count_is_rejected(self):
        with self.assertRaises(ValueError):
            overfit.pbo_cscv(np.ones((64, 2)), n_splits=3)

    def test_zero_variance_candidate_is_not_a_pbo_pass(self):
        matrix = np.column_stack([
            np.zeros(64),
            np.linspace(-0.01, 0.01, 64),
        ])
        result = overfit.pbo_cscv(matrix, n_splits=8)
        self.assertIsNone(result["pbo"])
        self.assertIn("zero-variance", result["note"])

    def test_excessive_cscv_partitions_fail_closed(self):
        matrix = np.arange(68 * 2, dtype=float).reshape(68, 2)
        result = overfit.pbo_cscv(matrix, n_splits=34)
        self.assertIsNone(result["pbo"])
        self.assertIn("resource limits", result["note"])


class MultipleTestingTests(unittest.TestCase):
    def test_benjamini_hochberg_rejects_small_pvalues(self):
        p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
        result = overfit.benjamini_hochberg(p, alpha=0.05)
        self.assertIn(0, result["rejected"])
        self.assertGreater(result["threshold"], 0.0)

    def test_bonferroni_threshold_grows_with_trials(self):
        self.assertLess(overfit.bonferroni_t_threshold(1), overfit.bonferroni_t_threshold(1000))

    def test_haircut_rejects_marginal_t(self):
        self.assertTrue(overfit.multiple_testing_haircut(3.0, 1)["survives"])
        self.assertFalse(overfit.multiple_testing_haircut(3.0, 100000)["survives"])

    def test_non_finite_pvalue_fails_closed(self):
        result = overfit.benjamini_hochberg([0.01, float("nan")])
        self.assertFalse(result["valid"])
        self.assertEqual(result["rejected"], [])

    def test_invalid_haircut_input_cannot_survive(self):
        result = overfit.multiple_testing_haircut(float("nan"), 2)
        self.assertFalse(result["valid"])
        self.assertFalse(result["survives"])


if __name__ == "__main__":
    unittest.main()
