# -*- coding: utf-8 -*-
"""
crypto_cdd_fetcher.py
-------------------------------------
从 CryptoDataDownload (CDD) 下载静态 CSV 数据。
支持 Binance 现货 1h/4h/1d 数据，
输出标准格式，可直接供 crypto_chip_timeline_analysis_PRO_v3.9.1.py 使用。

Usage:
  python crypto_cdd_fetcher.py --symbol ETHUSDT --freq 1h --out data/Binance_ETHUSDT_1h.csv
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
    if not freq.endswith("h") and not freq.endswith("d"):
        raise ValueError("Unsupported freq, must be one of: 1h, 4h, 1d")
    # Example: Binance_ETHUSDT_1h.csv
    return f"{base}/Binance_{sym}_{freq}.csv"

def main():
    args = parse_args()
    url = build_url(args.symbol, args.freq)
    print(f"[*] Fetching: {url}")

    resp = requests.get(url)
    if resp.status_code != 200:
        raise SystemExit(f"[X] Download failed: HTTP {resp.status_code}")

    text = resp.text
    # 跳过注释行
    lines = [ln for ln in text.splitlines() if not ln.startswith("#") and ln.strip()]
    csv_data = "\n".join(lines)
    df = pd.read_csv(StringIO(csv_data))

    # 标准化列名
    cols_map = {c.lower().strip(): c for c in df.columns}
    for name in ["date", "open", "high", "low", "close"]:
        if name not in cols_map:
            raise SystemExit(f"[X] Missing column: {name}")
    vol_col = None
    for cand in ["volume usdt", "volumeusd", "volume"]:
        if cand in cols_map:
            vol_col = cols_map[cand]
            break
    if not vol_col:
        raise SystemExit("[X] Missing volume column")

    df = df.rename(columns={
        cols_map["date"]: "datetime",
        cols_map["open"]: "open",
        cols_map["high"]: "high",
        cols_map["low"]: "low",
        cols_map["close"]: "close",
        vol_col: "volume"
    })

    # CryptoDataDownload 数据时间是降序的（最新在上）
    df = df.iloc[::-1].reset_index(drop=True)
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[✓] Saved -> {args.out}  rows={len(df)}")

if __name__ == "__main__":
    main()
