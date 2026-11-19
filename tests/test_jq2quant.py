"""Unit tests for standalone JoinQuant transpiler (jq2quant)."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from jq2quant import run_jq_strategy, transpile, transpile_file
from jq2quant.cli import main as cli_main

SAMPLE_JQ_CODE = '''
def initialize(context):
    set_benchmark("000300.SH")
    g.security = "000001.SZ"
    run_daily(rebalance, time="open")

def rebalance(context):
    order_target_percent(g.security, 0.8)

def handle_data(context, data):
    if g.security in data:
        order_target_percent(g.security, 0.5)
'''


class Jq2QuantTests(unittest.TestCase):
    def test_transpile_ast_and_syntax(self):
        translated = transpile(SAMPLE_JQ_CODE, strategy_class_name="MyTestStrategy")

        # 1. Must be valid compilable Python
        compiled = compile(translated, "<transpiled_test>", "exec")
        self.assertIsNotNone(compiled)

        # 2. Check structure
        self.assertIn("class MyTestStrategy(ts.Strategy):", translated)
        self.assertIn("def on_init(self, context: ts.Context) -> None:", translated)
        self.assertIn("def on_bar(self, context: ts.Context, data: ts.CurrentData) -> None:", translated)
        self.assertIn("def rebalance(self, context):", translated)

        # 3. Check rewrites
        self.assertIn("context.set_benchmark(\"000300.SH\")", translated)
        self.assertIn("context.g.security = \"000001.SZ\"", translated)
        self.assertIn("self.order_target_percent(context.g.security, 0.8)", translated)
        self.assertIn("ts.run_daily(self.rebalance, time=\"open\")", translated)

    def test_transpile_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = Path(tmp_dir) / "jq_strat.py"
            out_file = Path(tmp_dir) / "modern_strat.py"
            in_file.write_text(SAMPLE_JQ_CODE, encoding="utf-8")

            res = transpile_file(in_file, out_file)
            self.assertTrue(out_file.exists())
            self.assertEqual(res, out_file.read_text(encoding="utf-8"))
            self.assertIn("class TranslatedStrategy(ts.Strategy):", res)

    def test_runtime_adapter_and_execution(self):
        # Create small mock panel
        dates = ["20240102", "20240103"]
        records = []
        for d in dates:
            records.append({
                "ts_code": "000001.SZ",
                "trade_date": d,
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "vol": 100000.0,
                "amount": 1000000.0,
                "up_limit": 11.0,
                "down_limit": 9.0,
            })
        mock_df = pd.DataFrame(records)

        result = run_jq_strategy(
            code_or_file=SAMPLE_JQ_CODE,
            start_date="20240102",
            end_date="20240103",
            initial_cash=500_000.0,
            custom_data=mock_df,
        )

        self.assertIsNotNone(result)
        self.assertGreater(len(result.orders), 0)
        self.assertGreater(result.portfolio_values.iloc[0], 500000.0)

    def test_cli_transpile_and_run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = Path(tmp_dir) / "test_input.py"
            out_file = Path(tmp_dir) / "test_output.py"
            in_file.write_text(SAMPLE_JQ_CODE, encoding="utf-8")

            # Test transpile command
            ret = cli_main(["transpile", str(in_file), "-o", str(out_file)])
            self.assertEqual(ret, 0)
            self.assertTrue(out_file.exists())


if __name__ == "__main__":
    unittest.main()
