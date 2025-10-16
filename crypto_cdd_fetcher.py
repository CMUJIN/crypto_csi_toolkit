# -*- coding: utf-8 -*-
"""
crypto_cdd_fetcher.py
-------------------------------------
Fetch static OHLCV CSVs from CryptoDataDownload (CDD).
Automatically skips comments and non-tabular lines.
Compatible with Binance 1h / 4h / 1d data.
"""

import argparse
import re
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

# --- robust datetime cleaner ---
_dt_frac_re = re.compile(r"(\.\d+)(?=$)")  # strip trailing .### at end
def clean_datetime_str(x: str) -> str:
    s = str(x).strip()
    # unify separators
    s = s.replace("T", " ").replace("Z", "")
    # remove trailing fractional seconds like .000, .800, .123456
    s = _dt_frac_re.sub("", s)
    # if only date provided, append 00:00:00
    if len(s) == 10 and re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        s = s + " 00:00:00"
    # if format like YYYY/MM/DD, normalize to YYYY-MM-DD
    s = s.replace("/", "-")
    return s

def main():
    args = parse_args()
    url = build_url(args.symbol, args.freq)
    print(f"[*] Fetching: {url}")

    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise SystemExit(f"[X] Download failed: HTTP {resp.status_code}")

    # Filter out comment lines and junk headers
    lines = []
    started = False
    for ln in resp.text.splitlines():
        if ln.strip().startswith("#") or not ln.strip():
            continue
        # Detect real header row (contains "Date" and "Open")
        if "Date" in ln and "Open" in ln:
            started = True
        if started:
            lines.append(ln)
    if not lines:
        raise SystemExit("[X] No valid CSV content detected in response.")

    df = pd.read_csv(StringIO("\n".join(lines)), low_memory=False)

    # Normalize column names
    cols = {c.lower().strip(): c for c in df.columns}
    date_col  = next((cols[k] for k in cols if "date" in k or "timestamp" in k or "datetime" in k), None)
    open_col  = next((cols[k] for k in cols if "open" in k), None)
    high_col  = next((cols[k] for k in cols if "high" in k), None)
    low_col   = next((cols[k] for k in cols if "low" in k), None)
    close_col = next((cols[k] for k in cols if "close" in k), None)
    # prefer USDT volume; otherwise generic "volume" (but not BTC volume)
    vol_col   = next((cols[k] for k in cols if "volume usdt" in k), None)
    if vol_col is None:
        vol_col = next((cols[k] for k in cols if ("volume" in k and "btc" not in k)), None)

    if not all([date_col, open_col, high_col, low_col, close_col, vol_col]):
        raise SystemExit(f"[X] Missing required columns. Found: {list(df.columns)}")

    df = df.rename(columns={
        date_col: "datetime",
        open_col: "open",
        high_col: "high",
        low_col: "low",
        close_col: "close",
        vol_col: "volume"
    })

    # 🔧 key fix: normalize datetime strings (remove any fractional seconds like .000/.800/.123456)
    df["datetime"] = df["datetime"].astype(str).map(clean_datetime_str)

    # sort ascending and keep standard columns
    df = df.iloc[::-1].reset_index(drop=True)
    df = df[["datetime", "open", "high", "low", "close", "volume"]]

    # enforce numeric on OHLCV to avoid dtype issues
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[✓] Saved -> {args.out}  rows={len(df)}")

if __name__ == "__main__":
    main()
