"""Alphalens-Powered Institutional Factor Analysis Tear Sheet.

Provides institutional-grade factor diagnosis (IC/IR, quantile spreads,
decay curves, turnover and monotonicity tests) matching Quantopian / Alphalens standards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import alphalens as al
import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "calc_factor_metrics",
    "generate_factor_tear_sheet",
    "prepare_factor_data",
]


def prepare_factor_data(
    factor: pd.Series | pd.DataFrame,
    prices: pd.DataFrame,
    periods: tuple[int, ...] = (1, 5, 10),
    quantiles: int = 5,
    max_loss: float = 0.5,
) -> pd.DataFrame:
    """Format factor and price series into Alphalens clean factor data."""
    # Ensure factor is MultiIndex (date, asset) Series
    if isinstance(factor, pd.DataFrame):
        if "trade_date" in factor.columns and "ts_code" in factor.columns:
            f_col = [c for c in factor.columns if c not in {"trade_date", "ts_code"}][0]
            f_series = factor.set_index(["trade_date", "ts_code"])[f_col]
        else:
            f_series = factor.stack()
    else:
        f_series = factor.copy()

    # Normalize index date types
    if not isinstance(f_series.index.levels[0], pd.DatetimeIndex):
        f_series.index = f_series.index.set_levels(pd.to_datetime(f_series.index.levels[0]), level=0)

    p_df = prices.copy()
    if not isinstance(p_df.index, pd.DatetimeIndex):
        p_df.index = pd.to_datetime(p_df.index)

    return al.utils.get_clean_factor_and_forward_returns(
        factor=f_series,
        prices=p_df,
        periods=periods,
        quantiles=quantiles,
        max_loss=max_loss,
    )


def calc_factor_metrics(factor_data: pd.DataFrame) -> dict[str, Any]:
    """Calculate institutional factor metrics from Alphalens factor data."""
    # 1. Information Coefficient (IC) analysis
    ic_df = al.performance.factor_information_coefficient(factor_data)
    mean_ic = al.performance.mean_information_coefficient(factor_data)
    ic_std = ic_df.std()
    ic_ir = mean_ic / ic_std.replace(0, np.nan)

    # 2. Quantile returns
    mean_ret_by_q, _ = al.performance.mean_return_by_quantile(factor_data, by_date=False)
    q_names = [f"Q{i}" for i in range(1, len(mean_ret_by_q) + 1)]

    # 3. Monotonicity score (Spearman rank correlation of quantile returns for period 1)
    first_col = mean_ret_by_q.columns[0]
    ranks = np.arange(1, len(mean_ret_by_q) + 1)
    vals = mean_ret_by_q[first_col].values
    monotonicity, _ = stats.spearmanr(ranks, vals) if len(vals) > 1 else (0.0, 1.0)

    # 4. Long-short spread
    long_short = {}
    for col in mean_ret_by_q.columns:
        long_short[str(col)] = float(mean_ret_by_q[col].iloc[-1] - mean_ret_by_q[col].iloc[0])

    # 5. Autocorrelation (turnover proxy)
    rank_autocorr = al.performance.factor_rank_autocorrelation(factor_data)
    mean_autocorr = float(rank_autocorr.mean()) if not rank_autocorr.empty else 0.0

    return {
        "mean_ic": {str(k): round(float(v), 4) for k, v in mean_ic.items()},
        "ic_ir": {str(k): round(float(v), 4) for k, v in ic_ir.items()},
        "ic_std": {str(k): round(float(v), 4) for k, v in ic_std.items()},
        "quantile_returns": {
            str(col): {q: round(float(mean_ret_by_q.loc[i, col]), 6) for i, q in zip(mean_ret_by_q.index, q_names)}
            for col in mean_ret_by_q.columns
        },
        "long_short_spread": {k: round(v, 6) for k, v in long_short.items()},
        "monotonicity_score": round(float(monotonicity), 4) if pd.notna(monotonicity) else 0.0,
        "mean_rank_autocorrelation": round(mean_autocorr, 4),
    }


def generate_factor_tear_sheet(
    factor: pd.Series | pd.DataFrame,
    prices: pd.DataFrame,
    periods: tuple[int, ...] = (1, 5, 10),
    quantiles: int = 5,
    max_loss: float = 0.5,
    output_html: str | Path | None = None,
) -> dict[str, Any]:
    """Generate complete institutional factor tear sheet and optional HTML report."""
    factor_data = prepare_factor_data(factor, prices, periods=periods, quantiles=quantiles, max_loss=max_loss)
    metrics = calc_factor_metrics(factor_data)

    if output_html:
        html = _render_alphalens_html(metrics)
        out_path = Path(output_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        metrics["html_report"] = str(out_path)

    return metrics


def _render_alphalens_html(metrics: dict[str, Any]) -> str:
    """Render interactive HTML factor tear sheet with Chart.js."""
    q_data = list(metrics.get("quantile_returns", {}).values())[0] if metrics.get("quantile_returns") else {}
    q_labels = list(q_data.keys())
    q_vals = [v * 10000 for v in q_data.values()]  # in bps

    ic_vals = list(metrics.get("mean_ic", {}).values())
    ir_vals = list(metrics.get("ic_ir", {}).values())

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Alphalens 因子评价全息透视表 - Quant Research Kit</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
  .container {{ max-width: 1000px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 32px; border: 1px solid #334155; }}
  h1 {{ color: #38bdf8; margin-top: 0; display: flex; justify-content: space-between; align-items: center; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 24px 0; }}
  .card {{ background: #0f172a; padding: 16px; border-radius: 8px; border: 1px solid #334155; }}
  .card-label {{ font-size: 13px; color: #94a3b8; }}
  .card-val {{ font-size: 24px; font-weight: 700; color: #38bdf8; margin-top: 6px; font-family: monospace; }}
  .chart-box {{ background: #0f172a; border-radius: 8px; border: 1px solid #334155; padding: 20px; margin-top: 24px; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<div class="container">
  <h1><span>Alphalens 因子质量透视表 (Factor Tear Sheet)</span></h1>
  <div class="grid">
    <div class="card"><div class="card-label">1期 Mean IC</div><div class="card-val">{ic_vals[0] if ic_vals else 0}</div></div>
    <div class="card"><div class="card-label">1期 IC_IR</div><div class="card-val">{ir_vals[0] if ir_vals else 0}</div></div>
    <div class="card"><div class="card-label">分层单调性评分</div><div class="card-val">{metrics.get('monotonicity_score', 0)}</div></div>
  </div>
  <div class="chart-box">
    <h3 style="margin-top: 0; color: #e2e8f0;">分层收益单调性 (Quantile Returns, bps)</h3>
    <canvas id="qChart" height="90"></canvas>
  </div>
</div>
<script>
  new Chart(document.getElementById('qChart').getContext('2d'), {{
    type: 'bar',
    data: {{
      labels: {q_labels},
      datasets: [{{ label: '各分组前瞻收益率 (基点 bps)', data: {q_vals}, backgroundColor: '#38bdf8' }}]
    }},
    options: {{ responsive: true, scales: {{ y: {{ grid: {{ color: '#334155' }} }} }} }}
  }});
</script>
</body>
</html>
"""
