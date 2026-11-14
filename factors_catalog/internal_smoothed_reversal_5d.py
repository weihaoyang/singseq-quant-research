"""Internal candidate: smoothed five-session reversal for long-only research."""


def compute_factor(df):
    """Return a twenty-session mean of the negative five-session return."""
    panel = df.sort_values(["ts_code", "trade_date"]).copy()
    reversal = -panel.groupby("ts_code")["adj_close"].pct_change(
        5, fill_method=None
    )
    return reversal.groupby(panel["ts_code"]).transform(
        lambda values: values.rolling(20, min_periods=5).mean()
    )
