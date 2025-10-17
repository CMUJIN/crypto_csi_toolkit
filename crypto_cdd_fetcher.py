#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto CDD Fetcher — lite (remove taker columns)
------------------------------------------------
✓ 输出字段统一为 9 列（去掉 taker_base_vol / taker_quote_vol）
✓ CDC 与 Binance 自动兼容、合并与回补
✓ 仅导出 >= --start 的数据
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

    # CDD 顶部有 1 行注释
    raw = pd.read_csv(io.StringIO(r.text), skiprows=1)

    # 列名清洗
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
    if "Symbol" not in raw.columns and "symbol" in raw.columns:
        raw = raw.rename(columns={"symbol": "Symbol"})
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

    # 成交量（基础）：可能是 Volume / Volume ETH / Volume BTC / Volume (ETH) 等
    vol_base_col = None
    for cand in ["Volume", "Volume ETH", "Volume BTC", "Volume (ETH)", "volume"]:
        if cand in raw.columns:
            vol_base_col = cand
            break
    if vol_base_col is None:
        raw["Volume"] = np.nan
    else:
        raw["Volume"] = pd.to_numeric(raw[vol_base_col], errors="coerce")

    # 成交量（quote / USDT）：可能是 Volume USDT / VolumeUSD / Volume USD / QuoteVolume
    vol_quote_col = None
    for cand in ["Volume USDT", "VolumeUSD", "Volume USD", "volume usd", "QuoteVolume", "Quote Volume"]:
        if cand in raw.columns:
            vol_quote_col = cand
            break
    if vol_quote_col is None:
        raw["Volume USDT"] = np.nan
    else:
        raw["Volume USDT"] = pd.to_numeric(raw[vol_quote_col], errors="coerce")

    # tradecount
    if "tradecount" in raw.columns:
        raw["tradecount"] = pd.to_numeric(raw["tradecount"], errors="coerce").astype("Int64")
    elif "Trades" in raw.columns:
        raw["tradecount"] = pd.to_numeric(raw["Trades"], errors="coerce").astype("Int64")
    else:
        raw["tradecount"] = pd.array([pd.NA] * len(raw), dtype="Int64")

    # 数值类型
    for k in ["Open", "High", "Low", "Close"]:
        raw[k] = pd.to_numeric(raw[k], errors="coerce")

    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    out = pd.DataFrame(
        {
            "Unix": raw["Unix"],
            "Date": raw["Date"],
            "Symbol": raw["Symbol"],
            "Open": raw["Open"],
            "High": raw["High"],
            "Low": raw["Low"],
            "Close": raw["Close"],
            "Volume": raw["Volume"],
            "Volume USDT": raw["Volume USDT"],
            "tradecount": raw["tradecount"],
        }
    )
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
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["Unix"] = df["open_time"].astype("int64")
            df["Symbol"] = symbol.upper()

            # 数值
            for c in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")

            df = df.sort_values("Date")

            out = pd.DataFrame(
                {
                    "Unix": df["Unix"],
                    "Date": df["Date"],
                    "Symbol": df["Symbol"],
                    "Open": df["open"],
                    "High": df["high"],
                    "Low": df["low"],
                    "Close": df["close"],
                    "Volume": df["volume"],
                    "Volume USDT": df["quote_asset_volume"],
                    "tradecount": df["trades"],
                }
            )

            print(
                f"[+] Binance fetched {len(out)} rows, "
                f"from {out['Date'].min()} to {out['Date'].max()}"
            )
            return _ensure_columns(out)
        except Exception as e:
            print(f"[!] Binance fetch error @ {ep}: {e}")

    return pd.DataFrame(columns=SCHEMA)


# -------------------------------
# 3) Merge & heal
# -------------------------------
def merge_and_heal(df_cdd: pd.DataFrame, df_binance: pd.DataFrame) -> pd.DataFrame:
    if df_binance is None or df_binance.empty:
        print("[=] Binance 数据为空，保留 CDD")
        merged = df_cdd.copy()
    else:
        merged = pd.concat([df_cdd, df_binance], ignore_index=True)

    # 去重 + 排序
    merged = merged.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)

    # 回推 Volume / Volume USDT（若缺失且可计算）
    cls = pd.to_numeric(merged["Close"], errors="coerce")
    vb = pd.to_numeric(merged["Volume"], errors="coerce")
    vq = pd.to_numeric(merged["Volume USDT"], errors="coerce")

    # 缺基础量，用报价/收盘回推
    need_b = vb.isna() & vq.notna() & cls.notna() & (cls != 0)
    merged.loc[need_b, "Volume"] = (vq / cls)[need_b]

    # 缺报价量，用基础*收盘回推
    need_q = vq.isna() & vb.notna() & cls.notna()
    merged.loc[need_q, "Volume USDT"] = (vb * cls)[need_q]

    merged = _ensure_columns(merged)
    return merged


# -------------------------------
# 4) Gap check
# -------------------------------
def check_gaps(df: pd.DataFrame, freq: str):
    if df.empty:
        return
    dt_hours = df["Date"].diff().dt.total_seconds().div(3600)
    gap_threshold = {"1h": 1.5, "4h": 5, "1d": 26}.get(freq, 1.5)
    gaps = df.loc[dt_hours > gap_threshold, "Date"]
    if len(gaps) > 0:
        print(f"[!] 时间缺口 {len(gaps)}（示例前5）：")
        for g in gaps.head(5):
            print("   -", g)
    else:
        print("[✓] 时间序列连续")


# -------------------------------
# 5) Main
# -------------------------------
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
        cdd = fetch_cdd(args.symbol, args.freq)
        last_dt = cdd["Date"].max()
        now_utc = datetime.now(timezone.utc)

        # 若 CDD 不是最新，则尝试用 Binance 补齐到最新
        df_binance = pd.DataFrame()
        if (now_utc - last_dt).total_seconds() / 3600 > 1:
            df_binance = fetch_binance(args.symbol, BINANCE_INTERVALS[args.freq], last_dt)
        else:
            print("[=] CDD 已接近最新，无需 Binance 补齐")

        merged = merge_and_heal(cdd, df_binance)

        # 仅保留 >= start
        out_df = merged[merged["Date"] >= start_dt].reset_index(drop=True)
        print(
            f"[*] Filter >= {start_dt.date()}, rows={len(out_df)}, "
            f"first={out_df['Date'].min() if not out_df.empty else 'NA'}, "
            f"last={out_df['Date'].max() if not out_df.empty else 'NA'}"
        )

        check_gaps(out_df, args.freq)

        out_df.to_csv(args.out, index=False)
        print(f"[OK] Saved -> {args.out}  rows={len(out_df)}")

    except Exception as e:
        # 失败兜底：尽量保存过滤后的 CDD
        print(f"[X] 运行错误: {e}")
        try:
            cdd = cdd[cdd["Date"] >= start_dt].reset_index(drop=True)
            cdd = _ensure_columns(cdd)
            cdd.to_csv(args.out, index=False)
            print(f"[!] 已保存过滤后的 CDD 数据 -> {args.out} rows={len(cdd)}")
        except Exception as ee:
            print(f"[X] 兜底保存失败: {ee}")


if __name__ == "__main__":
    main()
