"""Internal real-data candidate for a slower long-only reversal signal."""


def compute_factor(df):
    """Return the negative twenty-session adjusted-close return by stock."""
    panel = df.sort_values(["ts_code", "trade_date"])
    return -panel.groupby("ts_code")["adj_close"].pct_change(
        20, fill_method=None
    )
