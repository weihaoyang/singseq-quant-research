from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from quant_sdk.catalog_factors import (
    compute_named_factor,
    compute_named_factors,
    factor_family,
    runnable_factor_names,
)
from quant_sdk.research import run_named_factor


def _panel() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=90)
    for code_index in range(12):
        base = 10.0 + code_index
        for day_index, day in enumerate(dates):
            close = base * (1.0 + 0.0015 * day_index + 0.002 * np.sin(day_index / 4 + code_index))
            rows.append({
                "ts_code": f"{code_index:06d}.SZ",
                "trade_date": day.strftime("%Y%m%d"),
                "list_date": "20100101",
                "open": close * 0.997,
                "high": close * 1.012,
                "low": close * 0.988,
                "close": close,
                "adj_close": close,
                "vol": 100000 + code_index * 100 + day_index,
                "amount": close * (100000 + code_index * 100 + day_index),
                "benchmark_ret": 0.0005 * np.cos(day_index / 7),
            })
    return pd.DataFrame(rows)


class CatalogFactorExecutionTests(unittest.TestCase):
    def test_capability_manifest_covers_long_history_without_overclaiming(self) -> None:
        manifest = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "quant_sdk"
                / "factor_capability_manifest.json"
            ).read_text(encoding="utf-8")
        )
        factors = manifest["factors"]
        alpha101 = {
            item["name"] for item in factors
            if item["name"].startswith("alpha101_")
        }
        alpha191 = {
            item["name"] for item in factors
            if item["name"].startswith("alpha191_")
        }
        self.assertEqual(len(alpha101), 101)
        self.assertEqual(len(alpha191), 187)
        self.assertEqual(
            {
                f"alpha191_{index:03d}"
                for index in range(1, 192)
            }.difference(alpha191),
            {
                "alpha191_030",
                "alpha191_069",
                "alpha191_165",
                "alpha191_183",
            },
        )
        history = {
            item["name"]: int(item.get("minimumHistory") or 0)
            for item in factors
        }
        self.assertEqual(history["alpha191_149"], 253)
        self.assertEqual(history["pta_vfi"], 262)
        self.assertTrue(all(item["description"] for item in factors))
        self.assertTrue(all("executionValidated" in item for item in factors))
        self.assertFalse(
            next(
                item for item in factors if item["name"] == "alpha191_149"
            )["executionValidated"]
        )

    def test_short_panel_does_not_claim_long_history_factors(self) -> None:
        frame = _panel()
        short = frame.groupby("ts_code", sort=False).head(2).reset_index(drop=True)
        names = runnable_factor_names(short)
        self.assertLess(len(names), 900)
        self.assertNotIn("pta_vfi", names)
        self.assertNotIn("return_120", names)

    def test_family_discovery_returns_only_the_requested_family(self) -> None:
        frame = _panel()
        alpha101 = runnable_factor_names(frame, families=["alpha101"])
        self.assertTrue(alpha101)
        self.assertTrue(all(
            factor_family(name) == "alpha101" for name in alpha101
        ))

        momentum = runnable_factor_names(frame, families=["momentum"])
        self.assertIn("price_momentum_20", momentum)
        self.assertTrue(all(
            factor_family(name) == "momentum" for name in momentum
        ))

    def test_static_capability_manifest_matches_dynamic_discovery(self) -> None:
        frame = _panel()
        dynamic = runnable_factor_names(frame, use_manifest=False)
        manifest = runnable_factor_names(frame, use_manifest=True)
        self.assertEqual(manifest, dynamic)

    def test_real_implemented_families_are_selectable_and_causal(self) -> None:
        frame = _panel()
        names = runnable_factor_names(frame)
        self.assertIn("price_momentum_20", names)
        self.assertIn("roc_20", names)
        self.assertIn("alpha_006", names)
        self.assertIn("return_20", names)
        self.assertIn("parkinson_20", names)
        self.assertIn("alpha360_CLOSE0", names)

        for name in (
            "price_momentum_20", "roc_20", "alpha_006",
            "alpha_linear_decay_return_10", "return_20", "parkinson_20",
        ):
            values = compute_named_factor(name, frame)
            self.assertEqual(len(values), len(frame))
            self.assertTrue(np.isfinite(values.to_numpy(dtype=float)).any(), name)
        alpha360 = compute_named_factor("alpha360_CLOSE0", frame)
        self.assertTrue(np.isfinite(alpha360.to_numpy(dtype=float)).any())
        self.assertIn("talib_sma", names)
        if "mytt_rsi" in names:
            self.assertTrue(
                np.isfinite(compute_named_factor("mytt_rsi", frame).to_numpy(dtype=float)).any()
            )
        if "pta_rsi" in names:
            self.assertTrue(
                np.isfinite(compute_named_factor("pta_rsi", frame).to_numpy(dtype=float)).any()
            )

    def test_metadata_without_formula_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "尚无可运行实现"):
            compute_named_factor("cashflow_profit_match_ttm", _panel())

    def test_external_factor_batch_is_deterministic_across_workers(self) -> None:
        frame = _panel()
        external = [
            name for name in runnable_factor_names(frame)
            if name.lower().startswith(("alpha101_", "alpha191_"))
        ][:8]
        if len(external) < 2:
            self.skipTest("Alpha101/Alpha191 adapters are not installed")
        with mock.patch.dict(os.environ, {"QUANT_FACTOR_WORKERS": "1"}):
            sequential = compute_named_factors(external, frame)
        with mock.patch.dict(os.environ, {"QUANT_FACTOR_WORKERS": "4"}):
            parallel = compute_named_factors(external, frame)
        with mock.patch.dict(os.environ, {"QUANT_FACTOR_WORKERS": "4"}):
            zero_means_default = compute_named_factors(
                external, frame, workers=0,
            )
        self.assertEqual(list(parallel.columns), external)
        np.testing.assert_allclose(
            sequential.to_numpy(dtype=float),
            parallel.to_numpy(dtype=float),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            parallel.to_numpy(dtype=float),
            zero_means_default.to_numpy(dtype=float),
            equal_nan=True,
        )

    def test_external_alpha_cross_sections_are_not_collapsed(self) -> None:
        random = np.random.default_rng(90210)
        dates = pd.bdate_range("2024-01-02", periods=120)
        rows = []
        for code_index in range(12):
            close = 20.0 + np.cumsum(random.normal(0.02, 0.4, len(dates)))
            volume = random.lognormal(12, 0.6, len(dates))
            for day_index, day in enumerate(dates):
                price = max(float(close[day_index]), 1.0)
                rows.append({
                    "ts_code": f"{code_index:06d}.SZ",
                    "trade_date": day.strftime("%Y%m%d"),
                    "open": price * 0.997,
                    "high": price * 1.012,
                    "low": price * 0.988,
                    "close": price,
                    "vol": float(volume[day_index]),
                    "amount": price * float(volume[day_index]),
                })
        frame = pd.DataFrame(rows)
        try:
            values = compute_named_factors(
                ["alpha101_001", "alpha101_007"],
                frame,
            )
        except (ImportError, RuntimeError, ValueError) as error:
            self.skipTest(f"Alpha101 adapter is unavailable: {error}")
        for name in values.columns:
            finite_dates = frame.assign(value=values[name]).groupby(
                "trade_date",
            )["value"].nunique(dropna=True)
            self.assertGreater(
                int((finite_dates > 1).sum()),
                0,
                f"{name} collapsed every finite cross-section",
            )

    def test_registry_factor_batch_uses_compiled_panel_path(self) -> None:
        try:
            import numba  # noqa: F401
        except ImportError:
            self.skipTest("Numba is not installed")
        from quant_sdk.catalog_factors import (
            _compute_registry_factor,
            _registry_formulas,
        )

        frame = _panel()
        frame["benchmark_close"] = 3000.0 + np.arange(len(frame)) / len(frame)
        names = sorted(_registry_formulas())[:16]
        expected = {
            name: _compute_registry_factor(name, frame)
            for name in names
        }
        with mock.patch.dict(os.environ, {
            "QUANT_REGISTRY_BULK_MIN": "16",
            "QUANT_REGISTRY_BULK_WORK": "1",
        }):
            batch = compute_named_factors(names, frame)
        self.assertEqual(list(batch.columns), names)
        for name in names:
            np.testing.assert_allclose(
                batch[name].to_numpy(dtype=float),
                expected[name].to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-7,
                atol=1e-7,
                err_msg=name,
            )

    def test_technical_factor_batch_matches_single_outputs(self) -> None:
        frame = _panel()
        available = runnable_factor_names(frame)
        technical: list[str] = []
        for prefix in ("pta_", "talib_", "mytt_", "ta_", "qtl_"):
            technical.extend([
                name for name in available
                if name.startswith(prefix)
            ][:2])
        if len(technical) < 2:
            self.skipTest("technical indicator adapters are not installed")
        singles = {
            name: compute_named_factor(name, frame)
            for name in technical
        }
        batch = compute_named_factors(technical, frame)
        self.assertEqual(list(batch.columns), technical)
        for name in technical:
            np.testing.assert_allclose(
                batch[name].to_numpy(dtype=float),
                singles[name].to_numpy(dtype=float),
                equal_nan=True,
            )

    def test_technical_factor_process_shards_are_deterministic(self) -> None:
        dates = pd.bdate_range("2010-01-04", periods=4000)
        rows = []
        for code_index in range(6):
            for day_index, day in enumerate(dates):
                close = (
                    10.0 + code_index
                    + 0.0015 * day_index
                    + 0.002 * np.sin(day_index / 4 + code_index)
                )
                rows.append({
                    "ts_code": f"{code_index:06d}.SZ",
                    "trade_date": day.strftime("%Y%m%d"),
                    "open": close * 0.997,
                    "high": close * 1.012,
                    "low": close * 0.988,
                    "close": close,
                    "vol": 100000 + code_index * 100 + day_index,
                    "amount": close * (100000 + code_index * 100 + day_index),
                })
        frame = pd.DataFrame(rows)
        available = runnable_factor_names(frame)
        technical = []
        for prefix in ("pta_", "talib_", "mytt_", "ta_", "qtl_"):
            technical.extend([
                name for name in available
                if name.startswith(prefix)
            ][:1])
        if len(technical) < 2:
            self.skipTest("technical indicator adapters are not installed")
        sequential = compute_named_factors(technical, frame, workers=1)
        parallel = compute_named_factors(technical, frame, workers=4)
        pd.testing.assert_frame_equal(
            sequential,
            parallel,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_named_factor_runs_through_canonical_local_engine(self) -> None:
        frame = _panel()
        with tempfile.TemporaryDirectory(prefix="catalog-factor-") as tmp:
            panel_path = Path(tmp) / "panel.csv"
            output_path = Path(tmp) / "factor.csv"
            frame.to_csv(panel_path, index=False)
            result = run_named_factor(
                factor_name="price_momentum_20", data_path=str(panel_path),
                start="20240102", end="20240507", out_csv=str(output_path),
                tearsheet_path=str(Path(tmp) / "tearsheet.html"), groups=3)
            self.assertEqual(result["status"], "LOCAL_UNVERIFIED")
            self.assertEqual(result["factor_name"], "price_momentum_20")
            self.assertEqual(result["execution_boundary"], "catalog_implementation_local")
            self.assertEqual(result["factor_rows"], len(frame))
            self.assertGreater(result["metrics"]["n_ic_observations"], 0)
            self.assertGreater(len(set(result["curves"]["ls"])), 1)
            exported = pd.read_csv(output_path)
            self.assertEqual(list(exported.columns), ["ts_code", "trade_date", "factor_value"])
            self.assertGreater(int(exported["factor_value"].notna().sum()), 0)


if __name__ == "__main__":
    unittest.main()
