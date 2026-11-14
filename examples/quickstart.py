"""Execute the complete research workflow on deterministic synthetic data.

Synthetic observations exercise the software and carry no investment evidence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from quant_sdk.research import run_research


def sample_panel() -> pd.DataFrame:
    random = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=180).strftime("%Y%m%d")
    stocks = []
    for number in range(64):
        close = 20 * np.exp(np.cumsum(random.normal(0, 0.012, len(dates))))
        opening = close * np.exp(random.normal(0, 0.002, len(dates)))
        volume = random.uniform(1_000_000, 3_000_000, len(dates))
        stocks.append(pd.DataFrame({
            "ts_code": f"DEMO{number:03d}", "trade_date": dates,
            "open": opening, "high": np.maximum(opening, close) * 1.01,
            "low": np.minimum(opening, close) * 0.99, "close": close,
            "vol": volume, "amount": close * volume, "adj_close": close,
            "list_date": "20100101", "st": False,
        }))
    return pd.concat(stocks, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/quickstart"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_path = Path(__file__).with_name("ret5.py")
    source = source_path.read_text(encoding="utf-8")
    frame = sample_panel()
    data = args.out / "panel.csv"
    frame.to_csv(data, index=False)
    started = time.perf_counter()
    result = run_research(
        factor_source=source, data_path=str(data),
        start=frame.trade_date.min(), end=frame.trade_date.max(),
        out_csv=str(args.out / "factor.csv"),
        tearsheet_path=str(args.out / "report.html"), with_novelty=True,
    )
    report_path = args.out / "report.html"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace("<body>", "<body><p><strong>SYNTHETIC DATA DEMO</strong> · Deterministic software demonstration, 64 virtual securities and 180 sessions.</p>", 1)
    report_path.write_text(report, encoding="utf-8")
    summary = {
        "data_kind": "SYNTHETIC_SOFTWARE_DEMO", "status": result["status"],
        "rows": len(frame), "stocks": frame.ts_code.nunique(),
        "dates": frame.trade_date.nunique(), "start": frame.trade_date.min(),
        "end": frame.trade_date.max(), "elapsed_seconds": time.perf_counter() - started,
        "factor_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "input_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "factor_csv_sha256": hashlib.sha256((args.out / "factor.csv").read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256((args.out / "report.html").read_bytes()).hexdigest(),
        "ic_mean": result.get("ic_mean"), "metrics": result.get("tearsheet_metrics"),
        "approximations": result.get("approximations"),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nOpen the local report at {args.out.resolve() / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
