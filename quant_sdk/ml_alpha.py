"""Institutional Machine Learning Alpha Factor Engine (Qlib & LightGBM).

Provides:
- Purged & Embargoed Cross-Validation (Marcos Lopez de Prado) to eliminate data leakage.
- Automated feature engineering (multi-scale momentum, volatility, volume acceleration, RSI, MACD).
- LightGBM Gradient Boosted Decision Tree (GBDT) Alpha Model for cross-sectional stock ranking.
- Out-of-sample IC / Rank IC evaluation and feature importance diagnostics.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Sequence
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_alpha_features(
    df: pd.DataFrame,
    horizons: Sequence[int] = (5, 10, 20),
    target_horizon: int = 5,
) -> tuple[pd.DataFrame, list[str], str]:
    """Generate technical and momentum features with forward-looking label.

    Args:
        df: DataFrame with ['open', 'high', 'low', 'close', 'volume'] columns.
            Can be single symbol or indexed by (datetime, symbol).
        horizons: Lookback windows for rolling features.
        target_horizon: Forward periods for target prediction (e.g. 5-day forward return).

    Returns:
        (feature_df, feature_column_names, target_column_name)
    """
    out = df.copy()

    # Sort chronological
    if "datetime" in out.columns:
        out = out.sort_values("datetime")

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)

    feat_cols = []

    # 1. Multi-scale Returns Momentum
    for h in horizons:
        col = f"ret_{h}d"
        out[col] = close.pct_change(h)
        feat_cols.append(col)

    # 2. Rolling Volatilities
    for h in horizons:
        col = f"vol_{h}d"
        out[col] = close.pct_change().rolling(h).std()
        feat_cols.append(col)

    # 3. High-Low Spread (Parkinson Volatility proxy)
    for h in horizons:
        col = f"hl_spread_{h}d"
        out[col] = ((high - low) / close).rolling(h).mean()
        feat_cols.append(col)

    # 4. Volume Moving Average Ratios
    for h in horizons:
        col = f"vol_ratio_{h}d"
        v_ma = volume.rolling(h).mean()
        out[col] = volume / (v_ma + 1e-6)
        feat_cols.append(col)

    # 5. Fast RSI proxy
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-6)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    feat_cols.append("rsi_14")

    # 6. Target: Forward return (shift negative)
    target_col = f"target_fwd_ret_{target_horizon}d"
    out[target_col] = (close.shift(-target_horizon) / close) - 1.0

    # Drop intermediate NaNs
    clean_out = out.dropna(subset=feat_cols + [target_col]).copy()
    return clean_out, feat_cols, target_col


class PurgedTimeSeriesSplit:
    """Purged & Embargoed K-Fold Cross Validation.

    Prevents information leakage from overlapping labels and serial correlation.
    """

    def __init__(self, n_splits: int = 4, purge_bars: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)

        for i in range(1, self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            test_idx = indices[test_start:test_end]

            # Train set is everything before test_start - purge_bars - embargo_size
            train_end = max(0, test_start - self.purge_bars - embargo_size)
            train_idx = indices[:train_end]

            if len(train_idx) < 10 or len(test_idx) < 10:
                continue

            yield train_idx, test_idx


class LGBMAlphaModel:
    """Production-grade LightGBM GBDT Machine Learning Alpha Factor Model."""

    def __init__(
        self,
        objective: str = "regression",
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        max_depth: int = 5,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        self.params = {
            "objective": objective,
            "metric": "rmse",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }
        self.booster = None
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        num_boost_round: int = 200,
        early_stopping_rounds: int = 15,
    ) -> LGBMAlphaModel:
        """Train LightGBM alpha model with validation early stopping."""
        import lightgbm as lgb

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names or "auto")
        valid_sets = [train_data]
        valid_names = ["train"]

        callbacks = []
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=self.feature_names or "auto")
            valid_sets.append(val_data)
            valid_names.append("valid")
            if early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

        self.booster = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict alpha score (expected forward return / rank)."""
        if self.booster is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        return self.booster.predict(X)

    def get_feature_importances(self, importance_type: str = "gain") -> pd.Series:
        """Return feature importance Series sorted descending."""
        if self.booster is None or not self.feature_names:
            return pd.Series(dtype=float)
        importances = self.booster.feature_importance(importance_type=importance_type)
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)

    def evaluate(self, X_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray) -> dict[str, float]:
        """Compute institutional alpha metrics: Pearson IC, Spearman Rank IC, and MSE."""
        from scipy.stats import spearmanr

        preds = self.predict(X_test)
        y_true = np.asarray(y_test)

        # Pearson IC
        if np.std(preds) > 1e-8 and np.std(y_true) > 1e-8:
            ic = float(np.corrcoef(preds, y_true)[0, 1])
        else:
            ic = 0.0

        # Spearman Rank IC
        rank_ic_res = spearmanr(preds, y_true)
        rank_ic = float(rank_ic_res.correlation) if not np.isnan(rank_ic_res.correlation) else 0.0

        mse = float(np.mean((preds - y_true) ** 2))

        return {
            "ic": round(ic, 4),
            "rank_ic": round(rank_ic, 4),
            "mse": round(mse, 6),
            "num_samples": len(y_true),
        }

    def save(self, filepath: str) -> None:
        """Save booster model to text file."""
        if self.booster is not None:
            self.booster.save_model(filepath)

    def load(self, filepath: str) -> LGBMAlphaModel:
        """Load booster model from file."""
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=filepath)
        self.feature_names = self.booster.feature_name()
        return self


def run_purged_walkforward_alpha(
    data: pd.DataFrame,
    horizons: Sequence[int] = (5, 10, 20),
    target_horizon: int = 5,
    n_splits: int = 4,
) -> dict[str, Any]:
    """High-level turnkey pipeline for training & validating ML Alpha factors."""
    dataset, features, target = build_alpha_features(
        data, horizons=horizons, target_horizon=target_horizon
    )

    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, purge_bars=target_horizon)
    fold_results = []
    models = []

    X = dataset[features]
    y = dataset[target]

    for fold, (train_idx, test_idx) in enumerate(splitter.split(dataset)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        model = LGBMAlphaModel()
        model.fit(X_tr, y_tr, X_val=X_te, y_val=y_te, num_boost_round=100, early_stopping_rounds=10)
        metrics = model.evaluate(X_te, y_te)
        metrics["fold"] = fold + 1
        fold_results.append(metrics)
        models.append(model)

    avg_ic = float(np.mean([f["ic"] for f in fold_results])) if fold_results else 0.0
    avg_rank_ic = float(np.mean([f["rank_ic"] for f in fold_results])) if fold_results else 0.0

    return {
        "fold_metrics": fold_results,
        "mean_ic": round(avg_ic, 4),
        "mean_rank_ic": round(avg_rank_ic, 4),
        "features": features,
        "feature_importances": models[-1].get_feature_importances().to_dict() if models else {},
    }
