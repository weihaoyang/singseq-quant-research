"""Institutional backtest tearsheet: extended metrics + a self-contained report.

The canonical engine already returns Sharpe/Calmar/ICIR/max-drawdown/win-rate,
the long-short curve, underwater, IC series and IC decay.  This module adds the
metrics a desk expects (Sortino, drawdown duration, factor autocorrelation,
style exposure, long-short spread, benchmark excess) and renders a portable
HTML + JSON report.  It is pure numpy/pandas/SVG — no chart-server dependency —
so a researcher can read and archive it locally.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["build", "render_html", "write_report"]

_ANN = 252.0


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _daily_returns(level: Optional[Sequence[float]]) -> np.ndarray:
    if level is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(level, dtype=float)
    if arr.size < 3:
        return np.asarray([], dtype=float)
    ret = np.diff(arr) / np.where(arr[:-1] == 0.0, np.nan, arr[:-1])
    return ret[np.isfinite(ret)]


def _sortino(rets: np.ndarray) -> Optional[float]:
    if rets.size < 3:
        return None
    downside = rets[rets < 0.0]
    dd = downside.std(ddof=1) if downside.size > 1 else 0.0
    if dd <= 0.0:
        return None
    return float(rets.mean() / dd * np.sqrt(_ANN))


def _max_dd_duration(underwater: Optional[Sequence[float]]) -> Optional[int]:
    if underwater is None:
        return None
    arr = np.asarray(underwater, dtype=float)
    if arr.size == 0:
        return None
    below = arr < -1e-9
    longest = current = 0
    for flag in below:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def _factor_autocorr(panel: pd.DataFrame, factor: Sequence[float]) -> Optional[float]:
    df = pd.DataFrame({
        "ts_code": panel["ts_code"].astype(str).to_numpy(),
        "trade_date": panel["trade_date"].astype(str).to_numpy(),
        "f": np.asarray(factor, dtype=float),
    })
    corrs = []
    for _, sub in df.groupby("ts_code", sort=False):
        sub = sub.sort_values("trade_date")
        a = sub["f"].to_numpy()
        if a.size < 3:
            continue
        x, y = a[:-1], a[1:]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            continue
        xx, yy = x[mask], y[mask]
        if xx.std() == 0 or yy.std() == 0:
            continue
        corrs.append(float(np.corrcoef(xx, yy)[0, 1]))
    return float(np.mean(corrs)) if corrs else None


def _style_exposure(panel: pd.DataFrame, factor: Sequence[float],
                    min_names: int = 20, max_dates: int = 120) -> Dict[str, float]:
    """Mean cross-sectional loading of the factor on each zoo baseline."""
    try:
        from evaluation import alpha_zoo
    except Exception:  # pragma: no cover
        return {}
    zoo = alpha_zoo.compute(panel)
    work = pd.DataFrame({
        "d": panel["trade_date"].astype(str).to_numpy(),
        "f": np.asarray(factor, dtype=float),
    })
    for name, series in zoo.items():
        work[name] = np.asarray(series, dtype=float)
    dates = [d for d, sub in work.groupby("d", sort=False) if len(sub) >= min_names]
    if len(dates) > max_dates:
        idx = np.linspace(0, len(dates) - 1, max_dates).astype(int)
        dates = [dates[i] for i in idx]
    cols = list(zoo)
    loadings: Dict[str, list] = {c: [] for c in cols}
    for d in dates:
        sub = work[work["d"] == d]
        y = sub["f"].rank(pct=True).to_numpy()
        X = sub[cols].rank(pct=True).to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if mask.sum() < min_names:
            continue
        yv, Xv = y[mask], X[mask]
        yv = (yv - yv.mean()) / (yv.std() or 1.0)
        Xv = (Xv - Xv.mean(0)) / np.where(Xv.std(0) == 0, 1.0, Xv.std(0))
        try:
            coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xv)), Xv]), yv, rcond=None)
        except np.linalg.LinAlgError:
            continue
        for i, c in enumerate(cols):
            loadings[c].append(abs(float(coef[i + 1])))
    return {c: round(float(np.mean(v)), 4) for c, v in loadings.items() if v}


def build(result: Any, panel: pd.DataFrame, factor: Sequence[float], *,
          n_groups: int = 5, n_trials: int = 1,
          benchmark_level: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Assemble the full tearsheet dictionary from an engine result + panel."""
    metrics = dict(getattr(result, "metrics", {}) or {})
    ls_curve = getattr(result, "ls_curve", None)
    rets = _daily_returns(ls_curve)
    ic = np.asarray(getattr(result, "ic_series", np.asarray([])), dtype=float)
    turnover = np.asarray(getattr(result, "turnover_series", np.asarray([])), dtype=float)
    ic_decay = getattr(result, "ic_decay", None)

    extended = {
        "sharpe": _finite(metrics.get("sharpe")),
        "sortino": _sortino(rets),
        "calmar": _finite(metrics.get("calmar")),
        "annualized_return": _finite(metrics.get("annualized")),
        "max_drawdown": _finite(metrics.get("maxdd")),
        "max_drawdown_duration": _max_dd_duration(getattr(result, "underwater", None)),
        "win_rate": _finite(metrics.get("winrate")),
        "ic_mean": _finite(metrics.get("ic_mean") if "ic_mean" in metrics else np.nanmean(ic) if ic.size else None),
        "icir": _finite(metrics.get("icir")),
        "ic_std": _finite(np.nanstd(ic)) if ic.size else None,
        "turnover_mean": _finite(metrics.get("turnover") if "turnover" in metrics else np.nanmean(turnover) if turnover.size else None),
        "factor_autocorr": _factor_autocorr(panel, factor),
        "n_groups": int(n_groups),
    }
    if ic_decay is not None:
        if isinstance(ic_decay, dict):
            decay_values = [float(v) for v in ic_decay.values()]
        else:
            decay_values = np.asarray(ic_decay, dtype=float).ravel().tolist()
        if decay_values:
            extended["ic_decay"] = [round(_finite(x) or 0.0, 5) for x in decay_values]

    # Deflated Sharpe Ratio: is the long-short Sharpe beyond what n_trials searches
    # would produce by chance?  Reported always; used as a scoring gate when set.
    if rets.size >= 3:
        try:
            from evaluation import overfit
            dsr = overfit.deflated_sharpe_ratio(rets, int(n_trials))
            extended["dsr"] = _finite(dsr.get("dsr"))
            extended["expected_max_sr"] = _finite(dsr.get("expected_max_sr"))
            extended["n_trials"] = int(n_trials)
        except Exception:                            # noqa: BLE001
            pass

    attribution = None
    try:
        from evaluation import attribution as _attribution
        attribution = _attribution.ic_attribution(panel, factor)
    except Exception:                                # noqa: BLE001
        attribution = None
    if benchmark_level is not None:
        b = _daily_returns(benchmark_level)
        extended["benchmark_annualized"] = _finite(b.mean() * _ANN) if b.size else None

    report = {
        "title": "Quant Local Backtest Tearsheet",
        "status": "LOCAL_UNVERIFIED",
        "engine": getattr(result, "acceleration_mode", "canonical"),
        "metrics": extended,
        "friction_breakdown": {k: round(float(v), 6) for k, v in (getattr(result, "friction_breakdown", {}) or {}).items()},
        "blocked_orders": dict(getattr(result, "blocked_orders", {}) or {}),
        "style_exposure": _style_exposure(panel, factor),
        "attribution": attribution,
        "curves": {
            "ls": [round(float(x), 6) for x in np.asarray(ls_curve, dtype=float)] if ls_curve is not None else [],
            "benchmark": [_finite(x) for x in np.asarray(result.benchmark_curve, dtype=float)]
            if getattr(result, "benchmark_curve", None) is not None else [],
            "groups": [[_finite(x) for x in group] for group in
                       np.asarray(result.group_curves, dtype=float)]
            if getattr(result, "group_curves", None) is not None else [],
            "gross": [_finite(x) for x in np.asarray(result.ls_curve_gross, dtype=float)]
            if getattr(result, "ls_curve_gross", None) is not None else [],
            "dates": [str(x) for x in panel["trade_date"].drop_duplicates().sort_values().tolist()]
            if ls_curve is not None else [],
            "drawdown": [round(float(x), 6) for x in np.asarray(getattr(result, "underwater", []), dtype=float)]
            if getattr(result, "underwater", None) is not None else [],
            "ic": [_finite(x) for x in ic],
            "turnover": [_finite(x) for x in turnover],
        },
    }
    return report


# --- rendering ------------------------------------------------------------

def _svg_curve(values: Sequence[float], width: int = 760, height: int = 200,
               color: str = "#2b6cb0") -> str:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if arr.size < 2 or not finite.any():
        return ""
    lo, hi = float(arr[finite].min()), float(arr[finite].max())
    span = hi - lo or 1.0
    step = width / (arr.size - 1)
    pts = " ".join(
        f"{i * step:.2f},{height - (float(v) - lo) / span * (height - 12) - 6:.2f}"
        for i, v in enumerate(arr) if np.isfinite(v))
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            f'preserveAspectRatio="none"><polyline fill="none" stroke="{color}" '
            f'stroke-width="1.6" points="{pts}"/></svg>')


_PERCENT_METRICS = {
    "annualized_return", "gross_annual_return", "benchmark_annualized",
    "max_drawdown", "turnover_mean", "win_rate",
}


def _display_value(value: Any, percent: bool = False) -> str:
    if value is None:
        return "—"
    if percent and isinstance(value, (int, float)):
        return f"{value:.2%}"
    return f"{value:.4f}" if isinstance(value, float) else html.escape(str(value))


def _rows(items: Dict[str, Any], *, metrics: bool = False) -> str:
    cells = []
    for key, value in items.items():
        shown = _display_value(value, metrics and key in _PERCENT_METRICS)
        cells.append(f"<tr><th>{html.escape(key)}</th><td>{shown}</td></tr>")
    return "\n".join(cells)


def render_html(report: Dict[str, Any], *, title: Optional[str] = None,
                subtitle: Optional[str] = None,
                curve_label: str = "策略净值（成本后）") -> str:
    t = html.escape(title or report.get("title", "Tearsheet"))
    sub = subtitle or "本机研究结果，正式比赛结果以赛季复核为准。"
    m = report.get("metrics", {})
    head = "".join(
        f"<div class='kpi'><span>{html.escape(k)}</span><strong>"
        f"{_display_value(m.get(k), k in _PERCENT_METRICS)}"
        f"</strong></div>"
        for k in ("sharpe", "sortino", "calmar", "annualized_return", "max_drawdown",
                  "win_rate", "ic_mean", "icir", "turnover_mean", "factor_autocorr"))
    series = report.get("curves", {})
    charts = []
    for key, label in (("ls", curve_label), ("gross", "策略净值（成本前）"), ("benchmark", "基准净值"),
                       ("drawdown", "回撤"), ("ic", "每日 IC"), ("turnover", "每日换手")):
        chart = _svg_curve(series.get(key, []))
        if chart:
            charts.append(f'<h2>{html.escape(label)}</h2><div class="chart">{chart}</div>')
    for index, values in enumerate(series.get("groups", [])):
        chart = _svg_curve(values)
        if chart:
            charts.append(f'<h2>第 {index + 1} 组净值（成本前）</h2><div class="chart">{chart}</div>')
    dates = series.get("dates", [])
    coverage_html = (
        f'<p>{html.escape(str(dates[0]))} 至 {html.escape(str(dates[-1]))} · {len(dates)} 个日期</p>'
        if dates else ""
    )
    provenance = report.get("provenance") or {}
    provenance_html = f"<h2>结果来源</h2><table>{_rows(provenance)}</table>" if provenance else ""
    unavailable = report.get("unavailable") or []
    unavailable_html = (
        "<p>未记录：" + html.escape("、".join(map(str, unavailable))) + "</p>"
        if unavailable else ""
    )
    configuration_html = ""
    if report.get("configuration"):
        configuration_html = f"<h2>回测配置</h2><table>{_rows(report['configuration'])}</table>"
    approximations_html = ""
    if report.get("approximations"):
        approximations_html = "<h2>数据假设与近似</h2><ul>" + "".join(
            f"<li>{html.escape(str(item))}</li>" for item in report["approximations"]) + "</ul>"
    decay = report.get("metrics", {}).get("ic_decay")
    decay_html = ""
    if decay:
        bars = "".join(
            "<div class='bar' style='height:0px' title='未记录'></div>" if v is None else
            f"<div class='bar' style='height:{max(2, int((1-abs(v))*80))}px' title='{v}'></div>"
            for v in map(_finite, decay)
        )
        decay_html = f"<h2>IC decay (h=1..{len(decay)})</h2><div class='bars'>{bars}</div>"
    attr = report.get("attribution") or {}
    attr_html = ""
    if isinstance(attr, dict) and attr:
        summary = {k: attr.get(k) for k in
                   ("ic", "residual_ic", "style_share", "residual_share", "dates_used")}
        contrib = {f"style:{k}": v
                   for k, v in (attr.get("style_contributions") or {}).items()}
        attr_html = (f"<h2>IC attribution (style vs residual)</h2>"
                     f"<table>{_rows({**summary, **contrib})}</table>")
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{t}</title><style>
body{{font:13px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1a2230;margin:24px;background:#f7f9fc}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#6b7a90;margin-bottom:16px}}
.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:12px 0}}
.kpi{{background:#fff;border:1px solid #e3e9f3;border-radius:10px;padding:8px 10px}}
.kpi span{{display:block;color:#8291a5;font-size:11px}} .kpi strong{{font-size:16px}}
table{{border-collapse:collapse;background:#fff;border:1px solid #e3e9f3;border-radius:10px;width:100%;margin:8px 0}}
th,td{{text-align:left;padding:4px 10px;border-bottom:1px solid #eef2f8;font-size:12px}}
.chart{{background:#fff;border:1px solid #e3e9f3;border-radius:10px;padding:8px}}
.bars{{display:flex;gap:3px;align-items:flex-end;height:90px}} .bar{{width:10px;background:#4c9be8;border-radius:2px}}
.tag{{display:inline-block;background:#fff3cd;color:#8a6d00;border-radius:6px;padding:1px 8px;font-size:11px}}
</style></head><body>
<h1>{t}</h1>
<div class="sub">引擎 {html.escape(str(report.get('engine','')))} · <span class="tag">{html.escape(str(report.get('status','')))}</span>
· {html.escape(sub)}</div>
<div class="kpis">{head}</div>
{coverage_html}
{configuration_html}
{approximations_html}
{"".join(charts)}
{decay_html}
<h2>Metrics</h2><table>{_rows({k: v for k, v in m.items() if k != 'ic_decay'}, metrics=True)}</table>
{provenance_html}
{unavailable_html}
<h2>Style exposure (|loading| vs zoo)</h2><table>{_rows(report.get('style_exposure', {}))}</table>
{attr_html}
<h2>Friction breakdown</h2><table>{_rows(report.get('friction_breakdown', {}))}</table>
<h2>Blocked orders</h2><table>{_rows(report.get('blocked_orders', {}))}</table>
</body></html>"""


def write_report(report: Dict[str, Any], html_path: str | Path, *,
                 title: Optional[str] = None) -> Path:
    path = Path(html_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report, title=title), encoding="utf-8")
    return path
