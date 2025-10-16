#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch crypto CSV from CryptoDataDownload (CDD) for Binance pairs.
Usage:
  python crypto_cdd_fetcher.py --symbol ETHUSDT --freq 1h --out data/Binance_ETHUSDT_1h_2025-08-01_to_latest.csv
"""

import argparse, sys, re
import pandas as pd
import requests
from io import StringIO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="Symbol, e.g. ETHUSDT")
    ap.add_argument("--freq", required=True, help="Frequency, e.g. 1h or d")
    ap.add_argument("--start", required=False, help="Start date, e.g. 2025-08-01")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    # --- 如果没提供 --start，就自动从文件名推断 ---
    if not args.start:
        m = re.search(r'_(\d{4}-\d{2}-\d{2})_to_', args.out)
        if m:
            args.start = m.group(1)
            print(f"[*] Auto-detected start date from filename: {args.start}")
        else:
            print("[X] Missing --start and could not infer from filename.")
            sys.exit(1)

    url = f"https://www.cryptodatadownload.com/cdd/Binance_{args.symbol}_{args.freq}.csv"
    print(f"[*] Fetching: {url}")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"[X] Download failed: {e}")
        sys.exit(1)

    csv_data = r.text.splitlines()
    csv_data = [line for line in csv_data if not line.startswith("Downloaded from")]
    df = pd.read_csv(StringIO("\n".join(csv_data)), low_memory=False)

    # detect date column
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        print(f"[X] No date/timestamp column found: {list(df.columns)}")
        sys.exit(2)

    df["datetime"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    # --- 过滤 start 日期之后的数据 ---
    start_dt = pd.to_datetime(args.start, utc=True)
    df = df[df["datetime"] >= start_dt]

    df.to_csv(args.out, index=False)
    print(f"[✓] Saved -> {args.out}  rows={len(df)}")

if __name__ == "__main__":
    main()
