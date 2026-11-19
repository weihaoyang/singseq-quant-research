"""jq2quant: Standalone AST & pattern-based JoinQuant-to-native Strategy Transpiler.

Translates legacy JoinQuant procedural strategies:
    def initialize(context):
        set_benchmark('000300.SH')
        g.stock = '000001.SZ'
        run_daily(rebalance, time='open')

    def handle_data(context, data):
        order_target_percent(g.stock, 0.5)

Into modern, decoupled, object-oriented Strategy classes:
    import quant_sdk as ts

    class TranslatedStrategy(ts.Strategy):
        def on_init(self, context: ts.Context) -> None:
            context.set_benchmark('000300.SH')
            context.g.stock = '000001.SZ'
            ts.run_daily(self.rebalance, time='open')

        def on_bar(self, context: ts.Context, data: ts.CurrentData) -> None:
            self.order_target_percent(context.g.stock, 0.5)
"""
from __future__ import annotations

import ast
import re
from pathlib import Path


class JoinQuantTranspiler:
    """Decoupled transpiler that converts JoinQuant source code into modern OOP Strategy code."""

    def __init__(self, strategy_class_name: str = "TranslatedStrategy"):
        self.class_name = strategy_class_name

    def transpile(self, source_code: str) -> str:
        """Transpile JoinQuant Python source into modern Python Strategy code."""
        # Validate syntax via AST first
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise ValueError(f"Input JoinQuant code contains syntax errors: {e}") from e

        # Analyze functions defined in source
        top_level_funcs = set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                top_level_funcs.add(node.name)

        lines = source_code.splitlines()

        header = [
            '"""Transpiled from JoinQuant strategy by jq2quant (https://github.com/weihaoyang/jq2quant).',
            "Strategy decoupled from proprietary APIs into clean modern Quant Strategy.",
            '"""',
            "from __future__ import annotations",
            "",
            "import quant_sdk as ts",
            "",
            f"class {self.class_name}(ts.Strategy):",
        ]

        body_lines: list[str] = []
        module_level_helpers: list[str] = []

        # State tracking
        current_func: str | None = None
        current_func_lines: list[str] = []

        def flush_function(f_name: str, f_lines: list[str]):
            # Rewrite signature
            if f_name == "initialize":
                sig = "    def on_init(self, context: ts.Context) -> None:"
            elif f_name == "before_trading_start":
                sig = "    def on_before_trading(self, context: ts.Context) -> None:"
            elif f_name == "handle_data":
                sig = "    def on_bar(self, context: ts.Context, data: ts.CurrentData) -> None:"
            else:
                # Custom helper function
                first_line = f_lines[0].strip()
                if "context" in first_line:
                    sig = re.sub(r"def\s+([a-zA-Z0-9_]+)\s*\((.*?)\):", r"    def \1(self, \2):", first_line)
                    # Normalize self duplicate if already present
                    sig = sig.replace("(self, self,", "(self,").replace("(self, self)", "(self)")
                else:
                    sig = re.sub(r"def\s+([a-zA-Z0-9_]+)\s*\((.*?)\):", r"    def \1(self, \2):", first_line)
                    sig = sig.replace("(self, self,", "(self,").replace("(self, self)", "(self)")

            transformed_body = []
            for b_line in f_lines[1:]:
                # Apply transformation rules on line
                line_t = self._transform_line(b_line, top_level_funcs)
                transformed_body.append("    " + line_t)

            return [sig] + transformed_body

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Detect top-level function definition
            match = re.match(r"^def\s+([a-zA-Z0-9_]+)\s*\(", line)
            if match:
                if current_func:
                    body_lines.extend(flush_function(current_func, current_func_lines))
                    body_lines.append("")
                current_func = match.group(1)
                current_func_lines = [line]
                i += 1
                continue

            if current_func:
                # If non-empty line has zero indentation and is not a comment, function ended
                if stripped and not line.startswith(" ") and not line.startswith("\t"):
                    body_lines.extend(flush_function(current_func, current_func_lines))
                    body_lines.append("")
                    current_func = None
                    # Fall through to process line as module level
                else:
                    current_func_lines.append(line)
                    i += 1
                    continue

            # Module-level imports or declarations outside functions
            if stripped.startswith("import ") or stripped.startswith("from "):
                # Skip duplicate or proprietary jqdata imports
                if "jqdata" in stripped or "jqlib" in stripped:
                    pass
                else:
                    module_level_helpers.append(line)
            elif stripped:
                module_level_helpers.append(line)
            i += 1

        if current_func:
            body_lines.extend(flush_function(current_func, current_func_lines))

        # Runner footer
        footer = [
            "",
            "",
            'if __name__ == "__main__":',
            "    # Standalone execution with zero proprietary platform dependency",
            "    result = ts.run_strategy(",
            f"        strategy={self.class_name}(),",
            '        start_date="20230101",',
            '        end_date="20231231",',
            "        initial_cash=1_000_000.0,",
            "    )",
            "    result.summary()",
            '    result.generate_html_report("backtest_report.html")',
            "",
        ]

        all_lines = []
        if module_level_helpers:
            all_lines.extend(module_level_helpers)
            all_lines.append("")
        all_lines.extend(header)
        all_lines.extend(body_lines)
        all_lines.extend(footer)

        return "\n".join(all_lines)

    def _transform_line(self, line: str, known_funcs: set[str]) -> str:
        """Transforms a single line of code inside a strategy function."""
        res = line

        # 1. g.<attr> -> context.g.<attr>
        res = re.sub(r"\bg\.([a-zA-Z0-9_]+)", r"context.g.\1", res)

        # 2. Top-level order calls -> self.order*
        order_funcs = [
            "order_target_percent",
            "order_target_value",
            "order_target",
            "order_value",
            "order",
            "cancel_order",
        ]
        for of in order_funcs:
            res = re.sub(rf"(?<![\.\w]){of}\s*\(", f"self.{of}(", res)

        # 3. Scheduled tasks: run_daily(func -> ts.run_daily(self.func
        for rf in ["run_daily", "run_weekly", "run_monthly"]:
            for kf in known_funcs:
                res = re.sub(rf"(?<![\.\w]){rf}\s*\(\s*{kf}\b", f"ts.{rf}(self.{kf}", res)

        # 4. Global settings: set_benchmark -> context.set_benchmark, set_commission -> ts.set_commission
        res = re.sub(r"(?<![\.\w])set_benchmark\s*\(", "context.set_benchmark(", res)
        res = re.sub(r"(?<![\.\w])set_commission\s*\(", "ts.set_commission(", res)
        res = re.sub(r"(?<![\.\w])set_slippage\s*\(", "ts.set_slippage(", res)
        res = re.sub(r"(?<![\.\w])set_option\s*\(", "ts.set_option(", res)

        # 5. Data query calls: get_fundamentals, query, attribute_history, history, get_price
        for qf in ["get_fundamentals", "query", "attribute_history", "history", "get_price", "get_current_data"]:
            res = re.sub(rf"(?<![\.\w]){qf}\s*\(", f"ts.{qf}(", res)

        # 6. Table namespaces: valuation, indicator, income, balance, cash_flow -> ts.valuation, etc.
        for tbl in ["valuation", "indicator", "income", "balance", "cash_flow"]:
            res = re.sub(rf"(?<![\.\w]){tbl}\.", f"ts.{tbl}.", res)

        return res


def transpile(source_code: str, strategy_class_name: str = "TranslatedStrategy") -> str:
    """Public helper to transpile JoinQuant code string."""
    return JoinQuantTranspiler(strategy_class_name).transpile(source_code)


def transpile_file(input_file: str | Path, output_file: str | Path | None = None) -> str:
    """Public helper to transpile a JoinQuant file to modern Strategy file."""
    in_path = Path(input_file)
    with open(in_path, encoding="utf-8") as f:
        src = f.read()
    translated = transpile(src)
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(translated)
    return translated
