"""Five-session price reversal, evaluated independently for each security."""
import pandas as pd


def compute_factor(df: pd.DataFrame) -> pd.Series:
    close = df.groupby("ts_code", sort=False)["close"]
    return -(df["close"] / close.shift(5) - 1.0)
