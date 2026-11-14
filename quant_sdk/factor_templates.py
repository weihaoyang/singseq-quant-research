"""Ready-made factor templates (Qlib Alpha158-style), runnable in the sandbox.

The catalogue exists so a researcher can start from a known-good factor instead of
a blank editor.  Every template is an ordinary ``compute_factor(df)`` body over the
canonical panel columns (ts_code / trade_date / open / high / low / close / vol /
amount) and passes the sandbox AST audit (no system, file or reflection access).

Families and formulas follow the published **Alpha158** feature set from
Microsoft **Qlib** (MIT) — https://github.com/microsoft/qlib — which is the
open-source reference for cross-sectional A-share factor research.  The templates
are re-expressed in plain pandas so they run with the toolkit's existing
dependencies and produce a canonical-engine result directly.
"""
from __future__ import annotations

from typing import Any

__all__ = ["TEMPLATES", "STARTER_NAMES", "as_manifest", "find"]

STARTER_NAMES = ("RET5", "MOM_WINDOW", "HL_RANGE")

#: name, family, source, code (the body is what the researcher edits in the client)
TEMPLATES: list[dict[str, str]] = [
    {
        "name": "RET5", "family": "price-reversal", "source": "Qlib Alpha158 RET (short reversal)",
        "title": "短期反转",
        "description": "比较股票最近5个交易日的涨跌，研究短期下跌后的后续表现。",
        "code": "def compute_factor(df):\n"
                "    close = df['close'].astype(float)\n"
                "    return -close.groupby(df['ts_code']).pct_change(5, fill_method=None)\n",
    },
    {
        "name": "REV_WINDOW", "family": "价格反转（可扫描）", "param": "WINDOW",
        "source": "Qlib Alpha158 RET family（参数化，供客户端参数扫描）",
        "code": "def compute_factor(df):\n"
                "    WINDOW = 5\n"
                "    close = df['close'].astype(float)\n"
                "    return -close.pct_change(WINDOW)\n",
    },
    {
        "name": "MOM_WINDOW", "family": "动量（可扫描）", "param": "WINDOW",
        "title": "20日动量",
        "description": "比较股票过去20个交易日的涨幅，观察强势股票的后续表现。",
        "source": "Qlib Alpha158 ROC family（参数化，供客户端参数扫描）",
        "code": "def compute_factor(df):\n"
                "    WINDOW = 20\n"
                "    close = df['close'].astype(float)\n"
                "    return close.groupby(df['ts_code']).pct_change(WINDOW, fill_method=None)\n",
    },
    {
        "name": "VOL_WINDOW", "family": "波动率（可扫描）", "param": "WINDOW",
        "source": "Qlib Alpha158 STD family（参数化，供客户端参数扫描）",
        "code": "def compute_factor(df):\n"
                "    WINDOW = 20\n"
                "    close = df['close'].astype(float)\n"
                "    return -close.pct_change().rolling(WINDOW).std()\n",
    },
    {
        "name": "RET20", "family": "price-reversal", "source": "Qlib Alpha158 RET (medium reversal)",
        "code": "def compute_factor(df):\n"
                "    close = df['close'].astype(float)\n"
                "    return -close.pct_change(20)\n",
    },
    {
        "name": "MOM20", "family": "price-momentum", "source": "Qlib Alpha158 ROC family",
        "code": "def compute_factor(df):\n"
                "    close = df['close'].astype(float)\n"
                "    return close.pct_change(20)\n",
    },
    {
        "name": "MA_GAP_5_20", "family": "moving-average", "source": "Qlib Alpha158 MA family",
        "code": "def compute_factor(df):\n"
                "    close = df['close'].astype(float)\n"
                "    return (close.rolling(5).mean() - close.rolling(20).mean()) / close\n",
    },
    {
        "name": "BIAS6", "family": "moving-average", "source": "Qlib Alpha158 BIAS",
        "code": "def compute_factor(df):\n"
                "    close = df['close'].astype(float)\n"
                "    ma = close.rolling(6).mean()\n"
                "    return (close - ma) / ma\n",
    },
    {
        "name": "RSI14", "family": "momentum", "source": "Qlib Alpha158 RSI",
        "code": "def compute_factor(df):\n"
                "    close = df['close'].astype(float)\n"
                "    diff = close.diff()\n"
                "    up = diff.clip(lower=0).rolling(14).mean()\n"
                "    down = (-diff.clip(upper=0)).rolling(14).mean()\n"
                "    return 100.0 - 100.0 / (1.0 + up / down.replace(0.0, float('nan')))\n",
    },
    {
        "name": "STD20", "family": "volatility", "source": "Qlib Alpha158 STD",
        "code": "def compute_factor(df):\n"
                "    close = df['close'].astype(float)\n"
                "    return -close.pct_change().rolling(20).std()\n",
    },
    {
        "name": "SKEW20", "family": "volatility", "source": "Qlib Alpha158 SKEW",
        "code": "def compute_factor(df):\n"
                "    close = df['close'].astype(float)\n"
                "    return -close.pct_change().rolling(20).skew()\n",
    },
    {
        "name": "KURT20", "family": "volatility", "source": "Qlib Alpha158 KURT",
        "code": "def compute_factor(df):\n"
                "    close = df['close'].astype(float)\n"
                "    return -close.pct_change().rolling(20).kurt()\n",
    },
    {
        "name": "HL_RANGE", "family": "price-shape", "source": "Qlib Alpha158 KLEN/KMID family",
        "title": "日内振幅",
        "description": "比较每日最高价与最低价的距离，研究价格波动与后续收益的关系。",
        "code": "def compute_factor(df):\n"
                "    close = df['close'].astype(float)\n"
                "    return (df['high'] - df['low']) / close\n",
    },
    {
        "name": "RSV20", "family": "price-shape", "source": "Qlib Alpha158 RSV",
        "code": "def compute_factor(df):\n"
                "    close = df['close'].astype(float)\n"
                "    lo = df['low'].rolling(20).min()\n"
                "    hi = df['high'].rolling(20).max()\n"
                "    return (close - lo) / (hi - lo).replace(0.0, float('nan'))\n",
    },
    {
        "name": "MAXRET20", "family": "price-shape", "source": "Qlib Alpha158 MAX (lottery reversal)",
        "code": "def compute_factor(df):\n"
                "    close = df['close'].astype(float)\n"
                "    return -(df['high'].rolling(20).max() / close - 1.0)\n",
    },
    {
        "name": "VMA_RATIO", "family": "volume", "source": "Qlib Alpha158 VMA / VSTD family",
        "code": "def compute_factor(df):\n"
                "    amount = df['amount'].astype(float)\n"
                "    return amount.rolling(5).mean() / amount.rolling(20).mean()\n",
    },
    {
        "name": "VOLUME_SURPRISE", "family": "volume", "source": "Qlib Alpha158 VROC family",
        "code": "def compute_factor(df):\n"
                "    vol = df['vol'].astype(float)\n"
                "    return vol / vol.rolling(20).mean()\n",
    },
    {
        "name": "ILLIQ20", "family": "liquidity", "source": "Amihud illiquidity (classic)",
        "code": "def compute_factor(df):\n"
                "    close = df['close'].astype(float)\n"
                "    amount = df['amount'].astype(float).replace(0.0, float('nan'))\n"
                "    return -(close.pct_change().abs() / (amount / 1e5)).rolling(20).mean()\n",
    },
    {
        "name": "CORR_AMT_CLOSE20", "family": "liquidity", "source": "Qlib Alpha158 CORR",
        "code": "def compute_factor(df):\n"
                "    close = df['close'].astype(float)\n"
                "    amount = df['amount'].astype(float)\n"
                "    return -close.rolling(20).corr(amount)\n",
    },
]


def as_manifest() -> list[dict[str, Any]]:
    """Compact form embedded in the public toolkit manifest (no login needed)."""
    return [{"name": t["name"], "family": t["family"], "source": t["source"],
             "code": t["code"], "param": t.get("param", ""),
             "title": t.get("title", t["name"]), "description": t.get("description", ""),
             "starter": t["name"] in STARTER_NAMES}
            for t in TEMPLATES]


def find(name: str) -> dict[str, str] | None:
    key = str(name or "").strip().upper()
    for item in TEMPLATES:
        if item["name"].upper() == key:
            return item
    return None


if __name__ == "__main__":  # pragma: no cover - convenience for humans/CI
    import json
    import sys

    if "--json" in sys.argv:
        print(json.dumps(as_manifest(), ensure_ascii=False, indent=2))
    else:
        for t in TEMPLATES:
            print(f"{t['name']:<20} {t['family']:<18} {t['source']}")
