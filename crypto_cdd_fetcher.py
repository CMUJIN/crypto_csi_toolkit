# -*- coding: utf-8 -*-
"""
crypto_cdd_fetcher.py
-------------------------------------
Fetch static OHLCV CSVs from CryptoDataDownload (CDD).
Supports Binance 1h / 4h / 1d files, outputs standardized ascending CSV.
"""

import argparse
import pandas as pd
import requests
from io import StringIO

def parse_args():
    ap = argparse.ArgumentParser(description="Fetch static OHLCV from CryptoDataDownload.")
    ap.add_argument("--symbol", required=True, help="Symbol, e.g. ETHUSDT")
    ap.add_argument("--freq", required=True, help="Frequency: 1h, 4h, or 1d")
    ap.add_argument("--out", required=True, help="Output CSV file path")
    return ap.parse_args()

def build_url(symbol: str, freq: str) -> str:
    base = "https://www.cryptodatadownload.com/cdd"
    sym = symbol.upper()
    freq = freq.lower()
    if not freq.endswith(("h", "d")):
        raise ValueError("Unsupported freq: must be one of 1h, 4h, 1d")
    return f"{base}/Binance_{sym}_{freq}.csv"

def main():
    args = parse_args()
    url = build_url(args.symbol, args.freq)
    print(f"[*] Fetching: {url}")

    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise SystemExit(f"[X] Download failed: HTTP {resp.status_code}")

    # Clean text and skip comments
    lines = [ln for ln in resp.text.splitlines() if ln.strip() and not ln.startswith("#")]
    csv_data = "\n".join(lines)

    df = pd.read_csv(StringIO(csv_data), low_memory=False)

    # Normalize column names (case-insensitive)
    cols = {c.lower().strip(): c for c in df.columns}

    # Detect essential columns
    date_col = next((cols[k] for k in cols if k in ["date", "timestamp", "datetime"]), None)
    open_col = next((cols[k] for k in cols if "open" in k), None)
    high_col = next((cols[k] for k in cols if "high" in k), None)
    low_col  = next((cols[k] for k in cols if "low" in k), None)
    close_col= next((cols[k] for k in cols if "close" in k), None)
    vol_col  = next((cols[k] for k in cols if "volume" in k and "btc" not in k), None)

    if not date_col:
        raise SystemExit(f"[X] No date/timestamp column found: {list(df.columns)}")
    if not vol_col:
        raise SystemExit("[X] Missing volume column")

    df = df.rename(columns={
        date_col: "datetime",
        open_col: "open",
        high_col: "high",
        low_col: "low",
        close_col: "close",
        vol_col: "volume"
    })

    # Sort ascending
    df = df.iloc[::-1].reset_index(drop=True)
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[✓] Saved -> {args.out}  rows={len(df)}")

if __name__ == "__main__":
    main()
