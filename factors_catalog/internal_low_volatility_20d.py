"""Internal real-data candidate for the official long-only gate.

The signal ranks stocks by the inverse of their recent adjusted-close
realized volatility. It is intentionally kept as a candidate until a real
Sandbox run confirms cost-adjusted performance and redundancy.
"""


def compute_factor(df):
    """Return the negative 20-session realized volatility by stock."""
    panel = df.sort_values(["ts_code", "trade_date"]).copy()
    returns = panel.groupby("ts_code")["adj_close"].pct_change(fill_method=None)
    volatility = returns.groupby(panel["ts_code"]).transform(
        lambda values: values.rolling(20, min_periods=20).std()
    )
    return -volatility
