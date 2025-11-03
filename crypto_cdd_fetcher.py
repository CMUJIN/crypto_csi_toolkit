#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto CDD Fetcher — extended for liquidity curve
-------------------------------------------------
✓ 输出字段统一为 9 列（去掉 taker_base_vol / taker_quote_vol）
✓ CDC 与 Binance 自动兼容、合并与回补
✓ 支持 --liquidity_start 参数：取 chip_start 与 liquidity_start 的最早时间作为下载起点
✓ Binance 451 自动切换镜像；失败不中断
"""

import argparse
import io
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CryptoCSI/lite)"}
BINANCE_INTERVALS = {"1h": "1h", "4h": "4h", "1d": "1d"}
SCHEMA = [
    "Unix", "Date", "Symbol",
    "Open", "High", "Low", "Close",
    "Volume", "Volume USDT",
    "tradecount",
]


# -------------------------------
# Utilities
# -------------------------------
def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in SCHEMA:
        if c not in df.columns:
            df[c] = np.nan
    return df[SCHEMA]


def _to_utc_ms(ts) -> int:
    if pd.isna(ts):
        return 0
    return int(pd.Timestamp(ts).tz_convert("UTC").timestamp() * 1000)


def _numeric(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# -------------------------------
# 1) Load & normalize CDC (CDD)
# -------------------------------
def fetch_cdd(symbol: str, freq: str) -> pd.DataFrame:
    url = f"https://www.cryptodatadownload.com/cdd/Binance_{symbol.upper()}_{freq}.csv"
    print(f"[*] Fetching CDD: {url}")
    r = requests.get(url, timeout=20, headers=HEADERS)
    if r.status_code != 200:
        raise ValueError(f"[X] Download failed: HTTP {r.status_code}")

    raw = pd.read_csv(io.StringIO(r.text), skiprows=1)
    raw.columns = [c.strip() for c in raw.columns]

    # 统一 Date
    if "Date" in raw.columns:
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", utc=True)
    elif "date" in raw.columns:
        raw["Date"] = pd.to_datetime(raw["date"], errors="coerce", utc=True)
    else:
        raise ValueError("[X] CDC CSV missing Date column")

    # Unix（如无则生成）
    if "Unix" not in raw.columns and "unix" in raw.columns:
        raw = raw.rename(columns={"unix": "Unix"})
    if "Unix" not in raw.columns:
        raw["Unix"] = raw["Date"].map(_to_utc_ms)

    # 统一 Symbol
    raw["Symbol"] = symbol.upper()

    # 统一 OHLC
    rename = {}
    for std, cands in {
        "Open": ["Open", "open"],
        "High": ["High", "high"],
        "Low": ["Low", "low"],
        "Close": ["Close", "close", "Last", "last", "CloseUSD", "closeusd"],
    }.items():
        for c in cands:
            if c in raw.columns:
                rename[c] = std
                break
    raw = raw.rename(columns=rename)

    # 成交量（基础）
    vol_base_col = None
    for cand in ["Volume", "Volume ETH", "Volume BTC", "Volume (ETH)", "volume"]:
        if cand in raw.columns:
            vol_base_col = cand
            break
    raw["Volume"] = pd.to_numeric(raw.get(vol_base_col, np.nan), errors="coerce")

    # 成交量（quote / USDT）
    vol_quote_col = None
    for cand in ["Volume USDT", "VolumeUSD", "Volume USD", "volume usd", "QuoteVolume", "Quote Volume"]:
        if cand in raw.columns:
            vol_quote_col = cand
            break
    raw["Volume USDT"] = pd.to_numeric(raw.get(vol_quote_col, np.nan), errors="coerce")

    # tradecount
    if "Trades" in raw.columns:
        raw["tradecount"] = pd.to_numeric(raw["Trades"], errors="coerce").astype("Int64")
    elif "tradecount" in raw.columns:
        raw["tradecount"] = pd.to_numeric(raw["tradecount"], errors="coerce").astype("Int64")
    else:
        raw["tradecount"] = pd.array([pd.NA] * len(raw), dtype="Int64")

    # 数值类型
    for k in ["Open", "High", "Low", "Close"]:
        raw[k] = pd.to_numeric(raw[k], errors="coerce")

    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    out = raw[["Unix", "Date", "Symbol", "Open", "High", "Low", "Close", "Volume", "Volume USDT", "tradecount"]]
    print(f"[✓] CDD normalized, rows={len(out)}, last={out['Date'].max()}")
    return _ensure_columns(out)


# -------------------------------
# 2) Load & normalize Binance
# -------------------------------
def _binance_endpoints():
    return [
        "https://api.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
    ]


def fetch_binance(symbol: str, interval: str, start_time: pd.Timestamp) -> pd.DataFrame:
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": 1000,
        "startTime": int(pd.Timestamp(start_time).timestamp() * 1000),
    }

    for ep in _binance_endpoints():
        try:
            r = requests.get(ep, params=params, timeout=12, headers=HEADERS)
            if r.status_code == 451:
                print(f"[!] Binance 451 @ {ep}, switching mirror...")
                continue
            if r.status_code != 200:
                print(f"[!] Binance fetch failed: HTTP {r.status_code} @ {ep}")
                continue

            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                print(f"[!] Binance returned empty @ {ep}")
                continue

            df = pd.DataFrame(
                data,
                columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore"
                ],
            )

            df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["Unix"] = df["open_time"].astype("int64")
            df["Symbol"] = symbol.upper()
            for c in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")

            out = pd.DataFrame({
                "Unix": df["Unix"], "Date": df["Date"], "Symbol": df["Symbol"],
                "Open": df["open"], "High": df["high"], "Low": df["low"],
                "Close": df["close"], "Volume": df["volume"],
                "Volume USDT": df["quote_asset_volume"], "tradecount": df["trades"]
            })

            print(f"[+] Binance fetched {len(out)} rows, from {out['Date'].min()} → {out['Date'].max()}")
            return _ensure_columns(out)
        except Exception as e:
            print(f"[!] Binance fetch error @ {ep}: {e}")

    return pd.DataFrame(columns=SCHEMA)


# -------------------------------
# 3) Merge & heal
# -------------------------------
def merge_and_heal(df_cdd: pd.DataFrame, df_binance: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([df_cdd, df_binance], ignore_index=True)
    merged = merged.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)

    cls = pd.to_numeric(merged["Close"], errors="coerce")
    vb = pd.to_numeric(merged["Volume"], errors="coerce")
    vq = pd.to_numeric(merged["Volume USDT"], errors="coerce")

    need_b = vb.isna() & vq.notna() & cls.notna() & (cls != 0)
    merged.loc[need_b, "Volume"] = (vq / cls)[need_b]

    need_q = vq.isna() & vb.notna() & cls.notna()
    merged.loc[need_q, "Volume USDT"] = (vb * cls)[need_q]

    return _ensure_columns(merged)


# -------------------------------
# 4) Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--freq", required=True, choices=["1h", "4h", "1d"])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--liquidity_start", required=False, help="Optional: earliest liquidity analysis start date")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ✅ 取最早的日期
    chip_start = pd.to_datetime(args.start, utc=True)
    liq_start = pd.to_datetime(args.liquidity_start, utc=True) if args.liquidity_start else chip_start
    start_dt = min(chip_start, liq_start)
    print(f"[*] Combined start = min(chip={chip_start.date()}, liquidity={liq_start.date() if liq_start else chip_start.date()}) -> {start_dt.date()}")

    try:
        cdd = fetch_cdd(args.symbol, args.freq)
        last_dt = cdd["Date"].max()
        now_utc = datetime.now(timezone.utc)

        df_binance = pd.DataFrame()
        if (now_utc - last_dt).total_seconds() / 3600 > 1:
            df_binance = fetch_binance(args.symbol, BINANCE_INTERVALS[args.freq], last_dt)
        else:
            print("[=] CDD 已接近最新，无需 Binance 补齐")

        merged = merge_and_heal(cdd, df_binance)
        out_df = merged[merged["Date"] >= start_dt].reset_index(drop=True)

        print(f"[*] Filter >= {start_dt.date()}, rows={len(out_df)}, first={out_df['Date'].min()}, last={out_df['Date'].max()}")
        out_df.to_csv(args.out, index=False)
        print(f"[OK] Saved -> {args.out} rows={len(out_df)}")

    except Exception as e:
        print(f"[X] Error: {e}")


if __name__ == "__main__":
    main()
