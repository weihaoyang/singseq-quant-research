"""Novelty baseline + novelty-weighted scoring + tearsheet contracts."""

import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evaluation import alpha_zoo, novelty, novelty_scoring, tearsheet


def _panel(codes: int = 40, days: int = 150, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=days).strftime("%Y%m%d").tolist()
    rows = []
    for ci in range(codes):
        code = f"{ci:06d}.SZ"
        price = 10.0 + rng.random() * 20.0
        for d in dates:
            price = max(1.0, price * (1.0 + rng.normal(0.0, 0.02)))
            high = price * (1.0 + abs(rng.normal(0, 0.01)))
            low = price * (1.0 - abs(rng.normal(0, 0.01)))
            rows.append({
                "ts_code": code, "trade_date": d,
                "open": price * (1 + rng.normal(0, 0.004)),
                "high": high, "low": low, "close": price,
                "vol": float(rng.integers(1_000, 50_000)),
                "amount": float(rng.integers(100_000, 5_000_000)),
            })
    return pd.DataFrame(rows)


class ZooTests(unittest.TestCase):
    def test_all_zoo_factors_compute(self):
        panel = _panel()
        zoo = alpha_zoo.compute(panel)
        self.assertEqual(sorted(zoo), alpha_zoo.zoo_names())
        for name, series in zoo.items():
            self.assertEqual(len(series), len(panel), name)
            self.assertTrue(np.isfinite(series.to_numpy()).any(), f"{name} is all-NaN")

    def test_zoo_version_is_frozen(self):
        self.assertEqual(alpha_zoo.ZOO_VERSION, "zoo-1")


class NoveltyTests(unittest.TestCase):
    def test_factor_equal_to_a_zoo_member_has_near_zero_novelty(self):
        panel = _panel()
        mom = alpha_zoo.compute(panel, ["momentum_20"])["momentum_20"]
        result = novelty.novelty(panel, mom)
        self.assertEqual(result.most_correlated, "momentum_20")
        self.assertGreater(result.max_corr, 0.95)
        self.assertLess(result.novelty, 0.1)

    def test_random_factor_is_more_novel(self):
        panel = _panel()
        rng = np.random.default_rng(11)
        random_factor = rng.normal(size=len(panel))
        base = novelty.novelty(panel, alpha_zoo.compute(panel, ["momentum_20"])["momentum_20"])
        other = novelty.novelty(panel, random_factor)
        self.assertGreater(other.novelty, base.novelty)

    def test_novelty_result_serialises(self):
        panel = _panel()
        result = novelty.novelty(panel, alpha_zoo.compute(panel, ["reversal_5"])["reversal_5"])
        payload = result.as_dict()
        self.assertEqual(payload["zoo_version"], "zoo-1")
        self.assertIn("top_correlated", payload)


class ScoringTests(unittest.TestCase):
    def test_gate_blocks_weak_factor(self):
        weak = {"ic_mean": 0.001, "icir": 0.01, "oos_retention": 0.1, "monotonicity": 0.2}
        self.assertFalse(novelty_scoring.passes_quality_gate(weak))
        score = novelty_scoring.final_score(weak, novelty=1.0)
        self.assertEqual(score["final_score"], 0.0)

    def test_novelty_premium_is_monotonic_and_bounded(self):
        strong = {"ic_mean": 0.05, "icir": 0.8, "oos_retention": 0.9, "monotonicity": 1.0}
        low = novelty_scoring.final_score(strong, novelty=0.0)
        high = novelty_scoring.final_score(strong, novelty=1.0)
        self.assertTrue(low["gate_passed"] and high["gate_passed"])
        self.assertGreater(high["final_score"], low["final_score"])
        # premium is confined to [w0, 1]
        cfg = novelty_scoring.ScoringConfig()
        self.assertAlmostEqual(low["novelty_premium"], cfg.w0, places=6)
        self.assertAlmostEqual(high["novelty_premium"], 1.0, places=6)

    def test_scoring_is_versioned(self):
        # bc0231c split one version string that used to describe two different
        # formulas: novelty scoring is "novelty-score-2" and season scoring has
        # its own "season-score-1" (server/services/season_scoring.py). This
        # assertion is the freeze: changing the formula without a deliberate edit
        # here (and, per the scoring-change gate, applying it to a new season
        # only) fails loudly.
        self.assertEqual(novelty_scoring.SCORING_VERSION, "novelty-score-2")
        payload = novelty_scoring.final_score(
            {"ic_mean": 0.05, "icir": 0.8, "oos_retention": 0.9, "monotonicity": 1.0}, 0.5)
        self.assertEqual(payload["scoring_version"], "novelty-score-2")
        self.assertIn("config", payload)


class TearsheetTests(unittest.TestCase):
    def _result(self, days: int = 120):
        rng = np.random.default_rng(3)
        ls = np.cumprod(1.0 + rng.normal(0.0005, 0.01, days))
        return SimpleNamespace(
            metrics={"sharpe": 1.2, "calmar": 0.8, "annualized": 0.15, "maxdd": -0.1,
                     "winrate": 0.55, "icir": 0.4},
            ls_curve=ls,
            underwater=np.minimum.accumulate(ls / np.maximum.accumulate(ls) - 1.0),
            ic_series=rng.normal(0.03, 0.05, days),
            turnover_series=np.abs(rng.normal(0.3, 0.05, days)),
            ic_decay=np.array([0.04, 0.03, 0.02, 0.01, 0.0]),
            friction_breakdown={"stamp": 1.0, "commission": 0.5, "slippage": 0.2},
            blocked_orders={"limit_up_buy": 2},
            acceleration_mode="cpu",
        )

    def test_build_and_render(self):
        panel = _panel()
        factor = alpha_zoo.compute(panel, ["reversal_5"])["reversal_5"].to_numpy()
        report = tearsheet.build(self._result(), panel, factor, n_groups=5)
        self.assertEqual(report["status"], "LOCAL_UNVERIFIED")
        self.assertIsNotNone(report["metrics"]["sortino"])
        self.assertIn("factor_autocorr", report["metrics"])
        self.assertTrue(report["style_exposure"])
        html = tearsheet.render_html(report)
        self.assertIn("Tearsheet", html)
        self.assertIn("LOCAL_UNVERIFIED", html)
        self.assertIn("<svg", html)

    def test_ic_decay_keeps_lags_and_uses_absolute_ic_height(self):
        panel = _panel()
        factor = alpha_zoo.compute(panel, ["reversal_5"])["reversal_5"].to_numpy()
        result = self._result()
        result.ic_decay = {1: -0.04, 2: 0.01, 5: 0.20}
        report = tearsheet.build(result, panel, factor, n_groups=5)
        self.assertEqual(report["metrics"]["ic_decay_lags"], [1, 2, 5])
        html = tearsheet.render_html(report)
        self.assertIn("IC decay (h=1,2,5)", html)
        self.assertIn("height:3px", html)
        self.assertIn("height:16px", html)

    def test_turnover_mean_excludes_the_terminal_no_trade_slot(self):
        panel = _panel()
        factor = alpha_zoo.compute(panel, ["reversal_5"])["reversal_5"].to_numpy()
        result = self._result(days=3)
        result.turnover_series = np.array([1.0, 3.0, 0.0])
        report = tearsheet.build(result, panel, factor, n_groups=5)
        self.assertEqual(report["metrics"]["turnover_mean"], 2.0)


class ResearchRunnerTests(unittest.TestCase):
    def test_scaffold_emits_a_compute_factor_template(self):
        from quant_sdk import research as _research
        src = _research.scaffold_factor("mom20", "momentum", 20)
        self.assertIn("def compute_factor(df:", src)
        self.assertIn("pct_change(20)", src)

    def test_local_backtest_reports_novelty_and_scoring(self):
        panel = _panel()
        from quant_sdk.local_backtest import run_local_backtest
        src = "def compute_factor(df):\n    return df.groupby('ts_code')['close'].pct_change(20)\n"
        dates = panel["trade_date"].nunique()
        out = run_local_backtest(src, panel, benchmark_ret=np.zeros(dates),
                                 with_novelty=True)
        self.assertEqual(out["status"], "LOCAL_UNVERIFIED")
        self.assertIn("novelty", out)
        self.assertIn(out["novelty"]["most_correlated"], ("momentum_20", "reversal_20"))
        self.assertGreater(out["novelty"]["max_corr"], 0.95)
        self.assertFalse(out["scoring"]["gate_passed"])
        self.assertEqual(out["scoring"]["final_score"], 0.0)


class BridgeAndRunnerTests(unittest.TestCase):
    @staticmethod
    def _package(root: Path) -> Path:
        pkg = root / "daily-research-test"
        pkg.mkdir(parents=True)
        (pkg / "stock_list.csv").write_text(
            "ts_code,source_code,name\n000001.SZ,000001,平安银行\n000002.SZ,000002,万科A\n",
            encoding="utf-8-sig")
        header = "日期,股票代码,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率"
        with zipfile.ZipFile(pkg / "equity_daily_raw.zip", "w") as zf:
            for code in ("000001", "000002"):
                lines = [header]
                for i, day in enumerate(("20230103", "20230104", "20230105")):
                    lines.append(f"{day},{code},{10 + i},{10.5 + i},{11 + i},{9.5 + i},1000,100000,0.1,0.05,0.5,0.01")
                zf.writestr(f"{code}.csv", "\n".join(lines))
        (pkg / "trade_calendar.csv").write_text(
            "cal_date,is_open\n20230103,1\n20230104,1\n20230105,1\n", encoding="utf-8-sig")
        return pkg

    def test_bridge_builds_staging_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = self._package(Path(tmp))
            from quant_sdk import research as _research
            bridge = _research.build_staging(str(pkg), str(Path(tmp) / "s.duckdb"))
            self.assertEqual(bridge["status"], "OK")
            self.assertGreater(bridge["daily_rows"], 0)

    def test_run_research_from_panel_csv(self):
        panel = _panel()
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "panel.csv"
            panel.to_csv(csv, index=False)
            from quant_sdk import research as _research
            src = "def compute_factor(df):\n    return df.groupby('ts_code')['close'].pct_change(20)\n"
            out = _research.run_research(factor_source=src, data_path=str(csv),
                                         start="20230102", end="20231231",
                                         out_csv=str(Path(tmp) / "factor.csv"),
                                         with_novelty=True)
            self.assertEqual(out["status"], "LOCAL_UNVERIFIED")
            self.assertIn("novelty", out)

    def test_capabilities_is_machine_readable(self):
        from quant_sdk import research as _research
        caps = _research.capabilities()
        self.assertIn("entrypoints", caps)
        self.assertIn("auto", caps["entrypoints"])
        self.assertEqual(caps["zoo_version"], "zoo-1")
        self.assertEqual(caps["reference_factors"]["count"], 241)
        self.assertIn("Python >= 3.11", caps["requirements"])
        self.assertEqual(
            caps["reference_factors"]["minute_published_through"],
            "20260410",
        )


if __name__ == "__main__":
    unittest.main()
