#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto CDD Fetcher v2.7 — 列名强一致 & 自愈计算版
------------------------------------------------
✓ 输出字段完全一致（11列）
✓ CDC/Binance 列名全面归一化
✓ 合并后自动回推缺失列（Volume / Volume USDT / taker_*）
✓ 仅保留 >= --start 的数据
✓ Binance 451 自动镜像；失败不中断
"""

import argparse, os, io, requests, pandas as pd, numpy as np
from datetime import datetime, timezone

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CryptoCSI/2.7)"}
BINANCE_INTERVALS = {"1h": "1h", "4h": "4h", "1d": "1d"}

SCHEMA = [
    "Unix", "Date", "Symbol",
    "Open", "High", "Low", "Close",
    "Volume", "Volume USDT",
    "tradecount", "taker_base_vol", "taker_quote_vol",
]

# ---------- 工具 ----------
def _to_utc_ts(dt: pd.Timestamp) -> int:
    if pd.isna(dt): return np.int64(0)
    return int(pd.Timestamp(dt).tz_convert("UTC").timestamp() * 1000)

def _num(x):
    try: return float(x)
    except: return np.nan

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in SCHEMA:
        if c not in df.columns:
            df[c] = np.nan
    return df[SCHEMA]

# ---------- 读取 CDC 并归一 ----------
def fetch_cdd(symbol: str, freq: str) -> pd.DataFrame:
    url = f"https://www.cryptodatadownload.com/cdd/Binance_{symbol}_{freq}.csv"
    print(f"[*] Fetching CDD: {url}")
    r = requests.get(url, timeout=20, headers=HEADERS)
    if r.status_code != 200:
        raise ValueError(f"[X] Download failed: HTTP {r.status_code}")

    # CDD 第一行为注释，跳过
    raw = pd.read_csv(io.StringIO(r.text), skiprows=1)
    # 原表头大小写/空格不一，先下沉
    cols_map = {c: c.strip() for c in raw.columns}
    raw = raw.rename(columns=cols_map)

    # Date / Unix
    if "Date" in raw.columns:
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", utc=True)
    elif "date" in raw.columns:
        raw["Date"] = pd.to_datetime(raw["date"], errors="coerce", utc=True)
    else:
        raise ValueError("[X] CDC CSV 缺少 Date 列")

    if "Unix" not in raw.columns:
        # 有些 CDC 不给 Unix，则用 Date 生成
        raw["Unix"] = raw["Date"].map(_to_utc_ts)

    # 核心 OHLC
    rename = {}
    for std, cands in {
        "Open": ["Open","open"],
        "High": ["High","high"],
        "Low":  ["Low","low"],
        "Close":["Close","close","Last","last","CloseUSD","closeusd"],
        "Symbol":["Symbol","symbol"],
    }.items():
        for c in cands:
            if c in raw.columns: rename[c] = std; break
    raw = raw.rename(columns=rename)

    # 成交量（基础）：CDC 可能写成 Volume / Volume ETH / Volume BTC / Volume (ETH)
    vol_base = None
    for cand in ["Volume","Volume ETH","Volume BTC","Volume (ETH)","volume"]:
        if cand in raw.columns:
            vol_base = cand; break
    if vol_base is None:
        raw["Volume"] = np.nan
    else:
        raw["Volume"] = pd.to_numeric(raw[vol_base], errors="coerce")

    # 成交量（报价）：Volume USDT / VolumeUSD / Volume USD
    vol_quote = None
    for cand in ["Volume USDT","VolumeUSD","Volume USD","volume usd","QuoteVolume","Quote Volume"]:
        if cand in raw.columns:
            vol_quote = cand; break
    if vol_quote is None:
        raw["Volume USDT"] = np.nan
    else:
        raw["Volume USDT"] = pd.to_numeric(raw[vol_quote], errors="coerce")

    # 笔数
    if "tradecount" in raw.columns:
        raw["tradecount"] = pd.to_numeric(raw["tradecount"], errors="coerce").astype("Int64")
    elif "Trades" in raw.columns:
        raw["tradecount"] = pd.to_numeric(raw["Trades"], errors="coerce").astype("Int64")
    else:
        raw["tradecount"] = pd.array([pd.NA]*len(raw), dtype="Int64")

    # taker
    if "taker_base_vol" in raw.columns:
        raw["taker_base_vol"] = pd.to_numeric(raw["taker_base_vol"], errors="coerce")
    elif "Taker buy base asset volume" in raw.columns:
        raw["taker_base_vol"] = pd.to_numeric(raw["Taker buy base asset volume"], errors="coerce")
    else:
        raw["taker_base_vol"] = np.nan

    if "taker_quote_vol" in raw.columns:
        raw["taker_quote_vol"] = pd.to_numeric(raw["taker_quote_vol"], errors="coerce")
    elif "Taker buy quote asset volume" in raw.columns:
        raw["taker_quote_vol"] = pd.to_numeric(raw["Taker buy quote asset volume"], errors="coerce")
    else:
        raw["taker_quote_vol"] = np.nan

    # 统一数值类型
    for k in ["Open","High","Low","Close"]:
        raw[k] = pd.to_numeric(raw[k], errors="coerce")

    raw["Symbol"] = symbol  # 强制一致
    raw = raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # 输出统一 schema
    out = pd.DataFrame({
        "Unix": raw["Unix"],
        "Date": raw["Date"],
        "Symbol": raw["Symbol"],
        "Open": raw["Open"],
        "High": raw["High"],
        "Low":  raw["Low"],
        "Close":raw["Close"],
        "Volume": raw["Volume"],
        "Volume USDT": raw["Volume USDT"],
        "tradecount": raw["tradecount"],
        "taker_base_vol": raw["taker_base_vol"],
        "taker_quote_vol": raw["taker_quote_vol"],
    })
    print(f"[✓] CDD normalized, rows={len(out)}, last={out['Date'].max()}")
    return _ensure_columns(out)


# ---------- 读取 Binance 并归一 ----------
def _binance_endpoints():
    return [
        "https://api.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
    ]

def fetch_binance(symbol: str, interval: str, start_time: pd.Timestamp) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": 1000,
              "startTime": int(pd.Timestamp(start_time).timestamp()*1000)}
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

            df = pd.DataFrame(data, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","quote_asset_volume","trades",
                "taker_buy_base","taker_buy_quote","ignore"
            ])
            df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["Unix"] = df["open_time"].astype("int64")
            df["Symbol"] = symbol

            # 数值转换
            num_cols = ["open","high","low","close","volume",
                        "quote_asset_volume","taker_buy_base","taker_buy_quote"]
            for c in num_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
            df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")

            df = df.sort_values("Date")
            out = pd.DataFrame({
                "Unix": df["Unix"],
                "Date": df["Date"],
                "Symbol": df["Symbol"],
                "Open": df["open"],
                "High": df["high"],
                "Low":  df["low"],
                "Close":df["close"],
                "Volume": df["volume"],
                "Volume USDT": df["quote_asset_volume"],
                "tradecount": df["trades"],
                "taker_base_vol": df["taker_buy_base"],
                "taker_quote_vol": df["taker_buy_quote"],
            })
            print(f"[+] Binance 补齐 {len(out)} 行, 起点 {out['Date'].min()}, 终点 {out['Date'].max()}")
            return _ensure_columns(out)
        except Exception as e:
            print(f"[!] Binance fetch error @ {ep}: {e}")
    return pd.DataFrame(columns=SCHEMA)

# ---------- 合并 & 自愈 ----------
def merge_and_heal(df_cdd: pd.DataFrame, df_binance: pd.DataFrame) -> pd.DataFrame:
    if df_binance is None or df_binance.empty:
        print("[=] Binance 数据为空，保留 CDD")
        merged = df_cdd.copy()
    else:
        merged = pd.concat([df_cdd, df_binance], ignore_index=True)

    # 去重 & 排序
    merged = merged.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)

    # 自愈：用 Close 推导 Volume / Volume USDT / taker_base_vol / taker_quote_vol
    close = pd.to_numeric(merged["Close"], errors="coerce")
    vol_b = pd.to_numeric(merged["Volume"], errors="coerce")
    vol_q = pd.to_numeric(merged["Volume USDT"], errors="coerce")
    tbb   = pd.to_numeric(merged["taker_base_vol"],  errors="coerce")
    tbq   = pd.to_numeric(merged["taker_quote_vol"], errors="coerce")

    # 1) 缺基础，用报价/收盘回推
    need_b = vol_b.isna() & vol_q.notna() & close.notna() & (close != 0)
    merged.loc[need_b, "Volume"] = (vol_q / close)[need_b]

    # 2) 缺报价，用基础*收盘回推
    need_q = vol_q.isna() & vol_b.notna() & close.notna()
    merged.loc[need_q, "Volume USDT"] = (vol_b * close)[need_q]

    # 3) taker_base 缺，用 taker_quote / Close 回推
    need_tbb = tbb.isna() & tbq.notna() & close.notna() & (close != 0)
    merged.loc[need_tbb, "taker_base_vol"] = (tbq / close)[need_tbb]

    # 4) taker_quote 缺，用 taker_base * Close 回推
    need_tbq = tbq.isna() & tbb.notna() & close.notna()
    merged.loc[need_tbq, "taker_quote_vol"] = (tbb * close)[need_tbq]

    # 统一列顺序
    merged = _ensure_columns(merged)
    return merged

# ---------- 缺口检测 ----------
def check_gaps(df: pd.DataFrame, freq: str):
    if df.empty: return
    dt = df["Date"].diff().dt.total_seconds().div(3600)
    gap_threshold = {"1h": 1.5, "4h": 5, "1d": 26}.get(freq, 1.5)
    gaps = df.loc[dt > gap_threshold, "Date"]
    if len(gaps) > 0:
        print(f"[!] 检测到 {len(gaps)} 个时间缺口（示例前5）：")
        for g in gaps.head(5): print("   -", g)
    else:
        print("[✓] 时间序列连续")

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--freq", required=True, choices=["1h","4h","1d"])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    start_dt = pd.to_datetime(args.start, utc=True)

    try:
        df_cdd = fetch_cdd(args.symbol, args.freq)
        last_dt = df_cdd["Date"].max()
        now_utc = datetime.now(timezone.utc)

        df_binance = pd.DataFrame()
        if (now_utc - last_dt).total_seconds()/3600 > 1:
            df_binance = fetch_binance(args.symbol, BINANCE_INTERVALS[args.freq], last_dt)
        else:
            print("[=] 数据接近最新，无需补齐")

        df_all = merge_and_heal(df_cdd, df_binance)
        df_out = df_all[df_all["Date"] >= start_dt].reset_index(drop=True)
        print(f"[*] Filter >= {start_dt.date()}, rows={len(df_out)}, "
              f"first={df_out['Date'].min() if not df_out.empty else 'NA'}, "
              f"last={df_out['Date'].max() if not df_out.empty else 'NA'}")

        check_gaps(df_out, args.freq)
        df_out.to_csv(args.out, index=False)
        print(f"[OK] Saved -> {args.out}  rows={len(df_out)}")

    except Exception as e:
        print(f"[X] 运行错误: {e}")
        try:
            df_cdd = df_cdd[df_cdd["Date"] >= start_dt].reset_index(drop=True)
            df_cdd = _ensure_columns(df_cdd)
            df_cdd.to_csv(args.out, index=False)
            print(f"[!] 已保存过滤后的 CDD 数据 -> {args.out} rows={len(df_cdd)}")
        except Exception as ee:
            print(f"[X] 兜底保存失败: {ee}")

if __name__ == "__main__":
    main()
