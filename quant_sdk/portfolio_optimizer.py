"""Institutional Portfolio Optimization module powered by PyPortfolioOpt.

Supports:
- Markowitz Modern Portfolio Theory (Mean-Variance): Max Sharpe, Min Volatility, Efficient Frontier
- Hierarchical Risk Parity (HRP) - Machine-learning / hierarchical clustering portfolio allocation
- Black-Litterman Asset Allocation - Combining market equilibrium with subjective alpha views
- Performance analytics: Expected annual return, annual volatility, Sharpe ratio.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _prepare_inputs(
    data: pd.DataFrame,
    is_prices: bool = True,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Helper to compute expected returns and sample/shrinkage covariance matrix.

    Returns:
        (mu, cov_matrix, returns_df)
    """
    from pypfopt import expected_returns, risk_models

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame of prices or returns.")

    clean_df = data.dropna(how="all").ffill().bfill()
    if clean_df.empty or clean_df.shape[1] < 2:
        raise ValueError("Data must have at least 2 assets and non-empty rows.")

    if is_prices:
        returns_df = clean_df.pct_change().dropna(how="all")
        mu = expected_returns.mean_historical_return(clean_df, frequency=252)
        # Ledoit-Wolf shrinkage covariance for numerical stability
        try:
            cov = risk_models.CovarianceShrinkage(clean_df, frequency=252).ledoit_wolf()
        except Exception:
            cov = risk_models.sample_cov(clean_df, frequency=252)
    else:
        returns_df = clean_df
        mu = returns_df.mean() * 252
        cov = returns_df.cov() * 252

    return mu, cov, returns_df


def optimize_max_sharpe(
    data: pd.DataFrame,
    risk_free_rate: float = 0.02,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    is_prices: bool = True,
    cutoff: float = 0.0001,
) -> dict[str, float]:
    """Calculate the Tangency portfolio that maximizes Sharpe Ratio.

    Args:
        data: Historical asset prices or returns (columns=symbols, index=datetime).
        risk_free_rate: Annualized risk-free rate (e.g. 0.02 for 2%).
        weight_bounds: Minimum and maximum weight for each asset (default: long-only [0, 1]).
        is_prices: True if data contains closing prices, False if returns.
        cutoff: Minimum threshold under which weights are set to 0.

    Returns:
        Dictionary mapping asset symbols to optimal portfolio weights.
    """
    from pypfopt import EfficientFrontier

    mu, cov, _ = _prepare_inputs(data, is_prices=is_prices)
    ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
    try:
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    except Exception as exc:
        logger.warning("Max Sharpe optimization encountered solver issue: %s. Falling back to Min Volatility.", exc)
        ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
        ef.min_volatility()

    cleaned_weights = ef.clean_weights(cutoff=cutoff)
    return dict(cleaned_weights)


def optimize_min_volatility(
    data: pd.DataFrame,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    is_prices: bool = True,
    cutoff: float = 0.0001,
) -> dict[str, float]:
    """Calculate the Global Minimum Variance (GMV) portfolio.

    Args:
        data: Historical asset prices or returns.
        weight_bounds: Min and max weight bounds.
        is_prices: True if price series.
        cutoff: Clean threshold.

    Returns:
        Asset weight dictionary.
    """
    from pypfopt import EfficientFrontier

    mu, cov, _ = _prepare_inputs(data, is_prices=is_prices)
    ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
    ef.min_volatility()
    return dict(ef.clean_weights(cutoff=cutoff))


def optimize_hrp(
    data: pd.DataFrame,
    is_prices: bool = True,
    cutoff: float = 0.0001,
) -> dict[str, float]:
    """Calculate Hierarchical Risk Parity (HRP) weights (Marcos Lopez de Prado).

    HRP uses graph theory and machine learning hierarchical tree clustering.
    It does NOT invert the covariance matrix and is impervious to singular or ill-conditioned matrices.
    """
    from pypfopt import HRPOpt

    _, _, returns_df = _prepare_inputs(data, is_prices=is_prices)
    hrp = HRPOpt(returns=returns_df)
    hrp.optimize()
    return dict(hrp.clean_weights(cutoff=cutoff))


def optimize_black_litterman(
    data: pd.DataFrame,
    market_caps: Mapping[str, float] | None = None,
    investor_views: Mapping[str, float] | None = None,
    view_confidences: list[float] | None = None,
    risk_free_rate: float = 0.02,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    is_prices: bool = True,
    cutoff: float = 0.0001,
) -> dict[str, float]:
    """Calculate optimal weights using the Black-Litterman model.

    Blends market equilibrium implied returns with explicit quantitative investor views.

    Args:
        data: Historical price DataFrame.
        market_caps: Dict of market capitalizations for universe assets (if None, equal-cap assumed).
        investor_views: Dict mapping symbol -> expected absolute annualized return (e.g. {'600519.SH': 0.15}).
        view_confidences: Confidence level for each view in [0.0, 1.0].
        risk_free_rate: Annual risk-free rate.
        weight_bounds: Long-only or leverage bounds.
        is_prices: True if price data.
        cutoff: Clean cutoff.
    """
    from pypfopt import BlackLittermanModel, EfficientFrontier, black_litterman

    _, cov, _ = _prepare_inputs(data, is_prices=is_prices)
    symbols = list(cov.columns)

    if market_caps is None:
        market_caps = {s: 1.0 / len(symbols) for s in symbols}
    else:
        # Fill missing with median
        med_cap = float(np.median(list(market_caps.values()))) if market_caps else 1.0
        market_caps = {s: float(market_caps.get(s, med_cap)) for s in symbols}

    delta = black_litterman.market_implied_risk_aversion(
        market_prices=None,
        risk_free_rate=risk_free_rate,
        cov_matrix=cov,
    ) if False else 2.5  # Standard risk aversion coefficient

    prior = black_litterman.market_implied_prior_returns(
        market_caps=market_caps,
        risk_aversion=delta,
        cov_matrix=cov,
        risk_free_rate=risk_free_rate,
    )

    if investor_views:
        # Filter views to assets in universe
        valid_views = {k: float(v) for k, v in investor_views.items() if k in symbols}
        if valid_views:
            bl = BlackLittermanModel(
                cov_matrix=cov,
                pi=prior,
                absolute_views=valid_views,
                omega="default" if not view_confidences else None,
                view_confidences=view_confidences,
            )
            rets = bl.bl_returns()
            cov_bl = bl.bl_cov()
        else:
            rets = prior
            cov_bl = cov
    else:
        rets = prior
        cov_bl = cov

    ef = EfficientFrontier(rets, cov_bl, weight_bounds=weight_bounds)
    try:
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    except Exception:
        ef.min_volatility()

    return dict(ef.clean_weights(cutoff=cutoff))


def evaluate_portfolio(
    weights: Mapping[str, float],
    data: pd.DataFrame,
    risk_free_rate: float = 0.02,
    is_prices: bool = True,
) -> dict[str, float]:
    """Compute annualized expected return, annual volatility, and Sharpe ratio for given weights.

    Args:
        weights: Dictionary mapping asset symbols to weights.
        data: Historical asset prices or returns.
        risk_free_rate: Risk-free rate.
        is_prices: True if prices.

    Returns:
        dict with keys: 'expected_return', 'volatility', 'sharpe_ratio'
    """
    mu, cov, _ = _prepare_inputs(data, is_prices=is_prices)
    w_vec = np.array([float(weights.get(s, 0.0)) for s in cov.columns])
    sum_w = np.sum(w_vec)
    if sum_w > 0:
        w_vec = w_vec / sum_w  # Normalize

    port_ret = float(np.dot(w_vec, mu))
    port_var = float(np.dot(w_vec.T, np.dot(cov, w_vec)))
    port_vol = float(np.sqrt(max(0.0, port_var)))
    sharpe = float((port_ret - risk_free_rate) / port_vol) if port_vol > 1e-8 else 0.0

    return {
        "expected_return": round(port_ret, 6),
        "volatility": round(port_vol, 6),
        "sharpe_ratio": round(sharpe, 4),
    }


def generate_portfolio_report_html(
    weights: Mapping[str, float],
    performance: Mapping[str, float],
    title: str = "Institutional Portfolio Allocation Report",
) -> str:
    """Generate a clean self-contained HTML report with asset weights and performance metrics."""
    filtered_weights = {k: v for k, v in weights.items() if v > 0.001}
    sorted_weights = sorted(filtered_weights.items(), key=lambda x: x[1], reverse=True)

    rows = "\n".join(
        f"<tr><td><strong>{sym}</strong></td><td>{wt * 100:.2f}%</td></tr>"
        for sym, wt in sorted_weights
    )

    labels = [s[0] for s in sorted_weights]
    values = [round(s[1] * 100, 2) for s in sorted_weights]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 24px; }}
    .card {{ background: #1e293b; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }}
    .stat-card {{ background: #334155; padding: 16px; border-radius: 6px; text-align: center; }}
    .stat-val {{ font-size: 24px; font-weight: bold; color: #38bdf8; }}
    .stat-lbl {{ font-size: 13px; color: #94a3b8; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #334155; text-align: left; }}
    th {{ color: #94a3b8; font-size: 13px; text-transform: uppercase; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="grid">
    <div class="stat-card">
      <div class="stat-val">{performance.get('expected_return', 0.0) * 100:.2f}%</div>
      <div class="stat-lbl">Expected Annual Return</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">{performance.get('volatility', 0.0) * 100:.2f}%</div>
      <div class="stat-lbl">Annualized Volatility</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">{performance.get('sharpe_ratio', 0.0):.2f}</div>
      <div class="stat-lbl">Sharpe Ratio</div>
    </div>
  </div>
  <div class="card">
    <h3>Optimal Weights Allocation</h3>
    <div style="max-height: 350px; display: flex; justify-content: center;">
      <canvas id="weightChart"></canvas>
    </div>
    <table>
      <thead><tr><th>Asset Symbol</th><th>Target Weight</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <script>
    const ctx = document.getElementById('weightChart').getContext('2d');
    new Chart(ctx, {{
      type: 'doughnut',
      data: {{
        labels: {labels},
        datasets: [{{
          data: {values},
          backgroundColor: ['#38bdf8', '#818cf8', '#c084fc', '#f472b6', '#fb7185', '#34d399', '#fbbf24', '#a3e635']
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ position: 'right', labels: {{ color: '#f8fafc' }} }} }}
      }}
    }});
  </script>
</body>
</html>"""
