#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto CDD Fetcher v2.6 — 完整字段对齐版
-----------------------------------------
✓ 输出字段完全一致（与 CDD）
✓ Binance 补齐数据字段完整（包含 taker 与 tradecount）
✓ 仅抓取 start 日期之后的数据
✓ 接口失败自动降级且不中断
"""

import argparse, os, io, requests, pandas as pd
from datetime import datetime, timezone

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CryptoCSI/2.6)"}
BINANCE_INTERVALS = {"1h": "1h", "4h": "4h", "1d": "1d"}


# ========== 1️⃣ CDD 下载 ==========
def fetch_cdd(symbol: str, freq: str) -> pd.DataFrame:
    url = f"https://www.cryptodatadownload.com/cdd/Binance_{symbol}_{freq}.csv"
    print(f"[*] Fetching CDD: {url}")
    r = requests.get(url, timeout=20, headers=HEADERS)
    if r.status_code != 200:
        raise ValueError(f"[X] Download failed: HTTP {r.status_code}")
    df = pd.read_csv(io.StringIO(r.text), skiprows=1)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.sort_values("Date").dropna(subset=["Date"]).reset_index(drop=True)
    print(f"[✓] CDD loaded, rows={len(df)}, last={df['Date'].max()}")
    return df


# ========== 2️⃣ Binance 下载 ==========
def _binance_endpoints():
    return [
        "https://api.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
    ]


def fetch_binance(symbol: str, interval: str, start_time: datetime) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": 1000,
              "startTime": int(start_time.timestamp() * 1000)}

    for ep in _binance_endpoints():
        try:
            r = requests.get(ep, params=params, timeout=12, headers=HEADERS)
            if r.status_code == 451:
                print(f"[!] Binance 451 @ {ep}, switching mirror...")
                continue
            if r.status_code != 200:
                print(f"[!] Binance fetch failed: HTTP {r.status_code}")
                continue

            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                print("[!] Binance returned empty")
                continue

            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])
            df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["Symbol"] = symbol
            df["Unix"] = df["open_time"]
            df = df.astype({
                "open": float, "high": float, "low": float,
                "close": float, "volume": float,
                "quote_asset_volume": float,
                "taker_buy_base": float, "taker_buy_quote": float,
                "trades": int
            })
            df = df.rename(columns={
                "open": "Open", "high": "High", "low": "Low", "close": "Close",
                "volume": "Volume", "quote_asset_volume": "Volume USDT",
                "trades": "tradecount", "taker_buy_base": "taker_base_vol",
                "taker_buy_quote": "taker_quote_vol"
            })
            df = df[[
                "Unix", "Date", "Symbol", "Open", "High", "Low", "Close",
                "Volume", "Volume USDT", "tradecount",
                "taker_base_vol", "taker_quote_vol"
            ]]
            print(f"[+] Binance 补齐 {len(df)} 行, 起点 {df['Date'].min()}, 终点 {df['Date'].max()}")
            return df
        except Exception as e:
            print(f"[!] Binance fetch error @ {ep}: {e}")
    return pd.DataFrame()


# ========== 3️⃣ 合并逻辑 ==========
def merge_data(df_cdd, df_binance):
    if df_binance.empty:
        print("[=] Binance 数据为空，使用 CDD 原始数据")
        return df_cdd
    merged = pd.concat([df_cdd, df_binance], ignore_index=True)
    merged = merged.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    print(f"[✓] 合并后总行数={len(merged)}, 最新={merged['Date'].max()}")
    return merged.reset_index(drop=True)


# ========== 4️⃣ 缺口检测 ==========
def check_gaps(df, freq):
    if freq not in ["1h", "4h", "1d"] or df.empty:
        return
    dt = df["Date"].diff().dt.total_seconds().div(3600)
    gap_threshold = {"1h": 1.5, "4h": 5, "1d": 26}[freq]
    gaps = df.loc[dt > gap_threshold, "Date"]
    if not gaps.empty:
        print(f"[!] 检测到 {len(gaps)} 个时间缺口（前5）:")
        for g in gaps.head(5):
            print("   ", g)
    else:
        print("[✓] 时间序列连续")


# ========== 5️⃣ 主流程 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--freq", required=True, choices=["1h", "4h", "1d"])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    start_dt = pd.to_datetime(args.start, utc=True)

    try:
        df_cdd = fetch_cdd(args.symbol, args.freq)
        last_dt = df_cdd["Date"].max()
        now_utc = datetime.now(timezone.utc)
        gap_hours = (now_utc - last_dt).total_seconds() / 3600

        df_binance = pd.DataFrame()
        if gap_hours > 1:
            df_binance = fetch_binance(args.symbol, BINANCE_INTERVALS[args.freq], last_dt)
        else:
            print("[=] 数据接近最新，无需补齐")

        df_all = merge_data(df_cdd, df_binance)
        df_final = df_all[df_all["Date"] >= start_dt].reset_index(drop=True)
        print(f"[*] Filter >= {start_dt.date()}, rows={len(df_final)}")

        check_gaps(df_final, args.freq)
        df_final.to_csv(args.out, index=False)
        print(f"[OK] Saved -> {args.out}")

    except Exception as e:
        print(f"[X] 运行错误: {e}")
        try:
            df_cdd = df_cdd[df_cdd["Date"] >= start_dt]
            df_cdd.to_csv(args.out, index=False)
            print(f"[!] 已保存过滤后的 CDD 数据 -> {args.out}")
        except Exception as ee:
            print(f"[X] 保存失败: {ee}")


if __name__ == "__main__":
    main()
