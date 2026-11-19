import unittest

import numpy as np
import pandas as pd

from quant_sdk.ml_alpha import (
    LGBMAlphaModel,
    PurgedTimeSeriesSplit,
    build_alpha_features,
    run_purged_walkforward_alpha,
)


class TestMLAlpha(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 250
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, size=n))
        high = close * (1 + np.abs(np.random.normal(0.005, 0.005, size=n)))
        low = close * (1 - np.abs(np.random.normal(0.005, 0.005, size=n)))
        open_ = (high + low) / 2
        volume = np.random.uniform(10000, 50000, size=n)

        self.df = pd.DataFrame(
            {"datetime": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
        )

    def test_build_alpha_features(self):
        feats_df, feat_cols, target_col = build_alpha_features(self.df, horizons=(5, 10), target_horizon=5)
        self.assertGreater(len(feat_cols), 5)
        self.assertIn("rsi_14", feat_cols)
        self.assertEqual(target_col, "target_fwd_ret_5d")
        self.assertFalse(feats_df.empty)

    def test_purged_split(self):
        splitter = PurgedTimeSeriesSplit(n_splits=3, purge_bars=5)
        splits = list(splitter.split(self.df))
        self.assertGreater(len(splits), 0)
        for tr_idx, te_idx in splits:
            # Check purging: tr_idx max should be < te_idx min
            self.assertLess(tr_idx[-1], te_idx[0])

    def test_lgbm_alpha_model_fit_predict(self):
        feats_df, feat_cols, target_col = build_alpha_features(self.df, horizons=(5, 10), target_horizon=5)
        split_idx = int(len(feats_df) * 0.7)
        train = feats_df.iloc[:split_idx]
        test = feats_df.iloc[split_idx:]

        model = LGBMAlphaModel()
        model.fit(
            train[feat_cols],
            train[target_col],
            X_val=test[feat_cols],
            y_val=test[target_col],
            num_boost_round=30,
            early_stopping_rounds=5,
        )

        preds = model.predict(test[feat_cols])
        self.assertEqual(len(preds), len(test))

        metrics = model.evaluate(test[feat_cols], test[target_col])
        self.assertIn("ic", metrics)
        self.assertIn("rank_ic", metrics)

        importances = model.get_feature_importances()
        self.assertFalse(importances.empty)

    def test_run_purged_walkforward_alpha(self):
        res = run_purged_walkforward_alpha(self.df, horizons=(5, 10), target_horizon=5, n_splits=3)
        self.assertIn("fold_metrics", res)
        self.assertIn("mean_ic", res)
        self.assertIn("mean_rank_ic", res)


if __name__ == "__main__":
    unittest.main()
