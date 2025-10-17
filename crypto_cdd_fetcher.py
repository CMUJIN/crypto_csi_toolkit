#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Data Downloader (v2.8-lite)
----------------------------------
✓ 去掉 taker_base_vol / taker_quote_vol 字段
✓ 保留 CDC + Binance 自动合并逻辑
✓ 自动过滤 start 日期以前的数据
✓ 接口失败不中断，打印日志提示
"""

import pandas as pd, numpy as np, requests, datetime as dt, os, sys, argparse

def fetch_cdc(symbol: str, freq: str):
    try:
        url = f"https://www.cryptodatadownload.com/cdd/Binance_{symbol.upper()}_{freq}.csv"
        df = pd.read_csv(url, skiprows=1)
        df["Date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.rename(columns={
            "unix": "Unix", "symbol": "Symbol",
            "open": "Open", "high": "High", "low": "Low", "close": "Close",
            "Volume USDT": "Volume USDT", "Volume": "Volume", "tradecount": "tradecount"
        })
        df = df[["Unix", "Date", "Symbol", "Open", "High", "Low", "Close", "Volume", "Volume USDT", "tradecount"]]
        print(f"[OK] CDD loaded, rows={len(df)}, last={df['Date'].max()}")
        return df
    except Exception as e:
        print(f"[!] CDC fetch failed: {e}")
        return pd.DataFrame()

def fetch_binance(symbol: str, freq: str, start_date: str):
    try:
        base = "https://api.binance.com/api/v3/klines"
        interval = freq.replace("m", "m").replace("h", "h").replace("d", "d")
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp.utcnow().timestamp() * 1000)
        limit = 1000
        rows = []
        while start_ts < end_ts:
            r = requests.get(base, params={"symbol": symbol.upper(), "interval": interval, "startTime": start_ts, "limit": limit}, timeout=10)
            if r.status_code != 200:
                print(f"[!] Binance fetch failed: HTTP {r.status_code}")
                break
            data = r.json()
            if not data:
                break
            rows.extend(data)
            start_ts = data[-1][0] + 1
            if len(data) < limit:
                break
        if not rows:
            return pd.DataFrame()
        cols = ["Unix", "Open", "High", "Low", "Close", "Volume", "_CloseTime", "_QuoteVolume", "_Trades", "_BaseTakerVol", "_QuoteTakerVol", "_Ignore"]
        df = pd.DataFrame(rows, columns=cols)
        df["Unix"] = df["Unix"].astype(np.int64)
        df["Date"] = pd.to_datetime(df["Unix"], unit="ms", utc=True)
        df["Symbol"] = symbol.upper()
        df["Volume USDT"] = pd.to_numeric(df["_QuoteVolume"], errors="coerce")
        df["tradecount"] = pd.to_numeric(df["_Trades"], errors="coerce")
        df = df[["Unix", "Date", "Symbol", "Open", "High", "Low", "Close", "Volume", "Volume USDT", "tradecount"]]
        print(f"[OK] Binance fetched, rows={len(df)}, last={df['Date'].max()}")
        return df
    except Exception as e:
        print(f"[!] Binance fetch error: {e}")
        return pd.DataFrame()

def merge_and_save(symbol, freq, start_date, out_path):
    cdc = fetch_cdc(symbol, freq)
    binance = fetch_binance(symbol, freq, start_date)

    df = binance if not binance.empty else cdc
    if df.empty:
        print(f"[X] No data for {symbol}")
        return

    start_dt = pd.to_datetime(start_date, utc=True)
    df = df[df["Date"] >= start_dt]
    df = df.sort_values("Date").drop_duplicates(subset=["Date"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved -> {out_path}, rows={len(df)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--freq", default="1h")
    ap.add_argument("--start", default="2025-08-01")
    ap.add_argument("--out", default="./data")
    args = ap.parse_args()

    out_file = os.path.join(args.out, f"Binance_{args.symbol}_{args.freq}_{args.start}_to_latest.csv")
    merge_and_save(args.symbol, args.freq, args.start, out_file)

if __name__ == "__main__":
    main()
