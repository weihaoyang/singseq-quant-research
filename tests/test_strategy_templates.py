"""Unit tests for Strategy Templates catalogue."""
import unittest

from quant_sdk.strategy_templates import (
    STRATEGY_TEMPLATES,
    get_strategy_template,
    list_strategy_templates,
)


class StrategyTemplatesTests(unittest.TestCase):
    def test_list_templates(self):
        items = list_strategy_templates()
        self.assertGreaterEqual(len(items), 5)
        ids = [item["id"] for item in items]
        self.assertIn("dual_ma", ids)
        self.assertIn("small_cap_pit_value", ids)
        self.assertIn("cb_double_low", ids)
        self.assertIn("etf_momentum_rotation", ids)
        self.assertIn("live_miniqmt_execution", ids)

    def test_get_template(self):
        tmpl = get_strategy_template("dual_ma")
        self.assertEqual(tmpl["title"], "双均线择时趋势跟踪")
        self.assertIn("import quant_sdk as ts", tmpl["code"])
        self.assertIn("run_strategy", tmpl["code"])

    def test_template_syntax(self):
        # Ensure all strategy templates parse without Python syntax errors
        for tmpl in STRATEGY_TEMPLATES:
            compile(tmpl["code"], f"<strategy_{tmpl['id']}>", "exec")


if __name__ == "__main__":
    unittest.main()
