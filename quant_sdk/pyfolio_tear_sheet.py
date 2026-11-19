"""PyFolio & Empyrical Institutional Risk & Portfolio Tear Sheet.

Provides hedge-fund level performance and risk breakdown:
- Rolling risk metrics (Sharpe, Sortino, Calmar, Omega, Tail ratio)
- Downside risk: VaR (95%), CVaR (Expected Shortfall), Skewness, Kurtosis
- Top 5 drawdown periods with exact peak, valley, recovery dates and duration
- Fama-French / Capital Asset Pricing Alpha and Beta against benchmark
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import empyrical as ep
import pandas as pd
import pyfolio as pf

__all__ = [
    "calc_pyfolio_risk_metrics",
    "generate_pyfolio_tear_sheet",
    "get_drawdown_table",
]


def calc_pyfolio_risk_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, Any]:
    """Calculate institutional performance & risk metrics using Empyrical."""
    r = returns.dropna()
    if len(r) < 2:
        return {}

    # Ensure datetime index
    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index)

    ann_ret = float(ep.annual_return(r))
    cum_ret = float(ep.cum_returns_final(r))
    ann_vol = float(ep.annual_volatility(r))
    downside_vol = float(ep.downside_risk(r))
    sharpe = float(ep.sharpe_ratio(r))
    sortino = float(ep.sortino_ratio(r))
    calmar = float(ep.calmar_ratio(r))
    omega = float(ep.omega_ratio(r))
    mdd = float(ep.max_drawdown(r))
    tail_r = float(ep.tail_ratio(r))
    var_95 = float(ep.value_at_risk(r, cutoff=0.05))
    cvar_95 = float(ep.conditional_value_at_risk(r, cutoff=0.05))

    # Benchmark-relative alpha and beta
    alpha, beta = 0.0, 1.0
    if benchmark_returns is not None and not benchmark_returns.empty:
        b = benchmark_returns.dropna()
        if not isinstance(b.index, pd.DatetimeIndex):
            b.index = pd.to_datetime(b.index)
        # Align dates
        aligned = pd.concat([r, b], axis=1, join="inner").dropna()
        if len(aligned) > 5:
            a_val, b_val = ep.alpha_beta(aligned.iloc[:, 0], aligned.iloc[:, 1])
            alpha = float(a_val) if pd.notna(a_val) else 0.0
            beta = float(b_val) if pd.notna(b_val) else 1.0

    return {
        "annual_return": round(ann_ret, 4),
        "cumulative_return": round(cum_ret, 4),
        "annual_volatility": round(ann_vol, 4),
        "downside_volatility": round(downside_vol, 4),
        "sharpe_ratio": round(sharpe, 2) if pd.notna(sharpe) else 0.0,
        "sortino_ratio": round(sortino, 2) if pd.notna(sortino) else 0.0,
        "calmar_ratio": round(calmar, 2) if pd.notna(calmar) else 0.0,
        "omega_ratio": round(omega, 2) if pd.notna(omega) else 0.0,
        "max_drawdown": round(mdd, 4),
        "tail_ratio": round(tail_r, 2) if pd.notna(tail_r) else 0.0,
        "var_95": round(var_95, 4),
        "cvar_95": round(cvar_95, 4),
        "alpha": round(alpha, 4),
        "beta": round(beta, 2),
    }


def get_drawdown_table(returns: pd.Series, top: int = 5) -> list[dict[str, Any]]:
    """Extract top N drawdown periods with peak, valley, recovery and duration."""
    r = returns.dropna()
    if len(r) < 2:
        return []
    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index)

    df_dd = pf.timeseries.gen_drawdown_table(r, top=top)
    records = []
    for _, row in df_dd.iterrows():
        val = row.get("Net drawdown in %")
        if pd.isna(val) or isinstance(val, type(pd.NaT)):
            continue
        p_date = row["Peak date"].strftime("%Y-%m-%d") if pd.notna(row["Peak date"]) else "-"
        v_date = row["Valley date"].strftime("%Y-%m-%d") if pd.notna(row["Valley date"]) else "-"
        r_date = row["Recovery date"].strftime("%Y-%m-%d") if pd.notna(row["Recovery date"]) else "未恢复"
        dur_val = row["Duration"]
        duration = int(dur_val) if pd.notna(dur_val) and not isinstance(dur_val, type(pd.NaT)) else -1
        records.append({
            "drawdown_pct": round(float(val), 2),
            "peak_date": p_date,
            "valley_date": v_date,
            "recovery_date": r_date,
            "duration_days": duration,
        })
    return records


def generate_pyfolio_tear_sheet(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    output_html: str | Path | None = None,
) -> dict[str, Any]:
    """Generate complete institutional portfolio tear sheet and optional HTML report."""
    metrics = calc_pyfolio_risk_metrics(returns, benchmark_returns)
    dd_table = get_drawdown_table(returns, top=5)
    metrics["top_drawdowns"] = dd_table

    if output_html:
        html = _render_pyfolio_html(returns, metrics, dd_table)
        out_path = Path(output_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        metrics["html_report"] = str(out_path)

    return metrics


def _render_pyfolio_html(returns: pd.Series, metrics: dict[str, Any], dd_table: list[dict[str, Any]]) -> str:
    """Render interactive HTML risk tear sheet with underwater curve."""
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max

    dates = [str(d)[:10] for d in drawdown.index]
    dd_values = [round(float(v) * 100, 2) for v in drawdown.values]

    dd_rows = "".join(
        f"<tr><td>-{d['drawdown_pct']}%</td><td>{d['peak_date']}</td><td>{d['valley_date']}</td>"
        f"<td>{d['recovery_date']}</td><td>{d['duration_days'] if d['duration_days'] > 0 else '进行中'} 天</td></tr>"
        for d in dd_table
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>PyFolio 机构级风险归因报告 - Quant Research Kit</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
  .container {{ max-width: 1100px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 32px; border: 1px solid #334155; }}
  h1 {{ color: #38bdf8; margin-top: 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
  .card {{ background: #0f172a; padding: 16px; border-radius: 8px; border: 1px solid #334155; }}
  .card-label {{ font-size: 13px; color: #94a3b8; }}
  .card-val {{ font-size: 22px; font-weight: 700; color: #38bdf8; margin-top: 6px; font-family: monospace; }}
  .red {{ color: #f87171; }}
  .green {{ color: #34d399; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 13px; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ background: #0f172a; color: #94a3b8; }}
  .chart-box {{ background: #0f172a; border-radius: 8px; border: 1px solid #334155; padding: 20px; margin-top: 24px; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<div class="container">
  <h1><span>PyFolio & Empyrical 投资组合全息风险透视表</span></h1>
  <div class="grid">
    <div class="card"><div class="card-label">年化收益 (CAGR)</div><div class="card-val green">+{metrics.get('annual_return', 0)*100:.2f}%</div></div>
    <div class="card"><div class="card-label">年化波动率</div><div class="card-val">{metrics.get('annual_volatility', 0)*100:.2f}%</div></div>
    <div class="card"><div class="card-label">夏普比率 (Sharpe)</div><div class="card-val">{metrics.get('sharpe_ratio', 0)}</div></div>
    <div class="card"><div class="card-label">索提诺比率 (Sortino)</div><div class="card-val">{metrics.get('sortino_ratio', 0)}</div></div>
    <div class="card"><div class="card-label">最大回撤 (MaxDD)</div><div class="card-val red">{metrics.get('max_drawdown', 0)*100:.2f}%</div></div>
    <div class="card"><div class="card-label">日度在险价值 (VaR 95%)</div><div class="card-val red">{metrics.get('var_95', 0)*100:.2f}%</div></div>
    <div class="card"><div class="card-label">期望损失 (CVaR 95%)</div><div class="card-val red">{metrics.get('cvar_95', 0)*100:.2f}%</div></div>
    <div class="card"><div class="card-label">Beta / Alpha</div><div class="card-val">{metrics.get('beta', 1.0)} / {metrics.get('alpha', 0)}</div></div>
  </div>

  <div class="chart-box">
    <h3 style="margin-top: 0; color: #e2e8f0;">水下回撤曲线 (Underwater Plot, %)</h3>
    <canvas id="underwaterChart" height="90"></canvas>
  </div>

  <h3 style="color: #e2e8f0; margin-top: 32px;">历史前 5 大深幅回撤期 (Worst Drawdown Periods)</h3>
  <table>
    <thead>
      <tr><th>回撤深度</th><th>最高点日期</th><th>谷底日期</th><th>恢复日期</th><th>持续天数</th></tr>
    </thead>
    <tbody>{dd_rows}</tbody>
  </table>
</div>
<script>
  new Chart(document.getElementById('underwaterChart').getContext('2d'), {{
    type: 'line',
    data: {{
      labels: {dates},
      datasets: [{{ label: '水下回撤深度 (%)', data: {dd_values}, borderColor: '#f87171', backgroundColor: 'rgba(248, 113, 113, 0.2)', fill: true, tension: 0.1, pointRadius: 0 }}]
    }},
    options: {{ responsive: true, scales: {{ y: {{ max: 0, grid: {{ color: '#334155' }} }} }} }}
  }});
</script>
</body>
</html>
"""
