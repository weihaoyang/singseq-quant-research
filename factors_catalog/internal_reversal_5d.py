"""Internal operator candidate used to exercise the real replay lifecycle.

This is a research candidate, not a published factor. It uses only the
point-in-time feature frame supplied by the canonical worker and reverses the
five-session adjusted-close return. Sandbox and Official results remain
subject to their respective evidence and publication gates.
"""


def compute_factor(df):
    """Return a five-session reversal score for each canonical row."""
    return -df.groupby("ts_code")["adj_close"].pct_change(5)
