#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Crypto Data Fetcher (v2.3-stable)
-------------------------------------------
✓ 下载 CDD 历史数据
✓ 自动检测最新日期并补齐 Binance 实时数据（UTC）
✓ Binance 接口异常时不中断执行，仅打印日志
✓ 自动检测数据连续性
✓ 保持原逻辑兼容：输出路径、参数、文件名一致
"""

import argparse, os, io, requests, pandas as pd
from datetime import datetime, timedelta, timezone

BINANCE_INTERVALS = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


# =========================
# 核心函数
# =========================
def fetch_cdd(symbol: str, freq: str) -> pd.DataFrame:
    """从 CryptoDataDownload 获取基础数据"""
    url = f"https://www.cryptodatadownload.com/cdd/Binance_{symbol}_{freq}.csv"
    print(f"[*] Fetching: {url}")
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        raise ValueError(f"[X] Download failed: HTTP {r.status_code}")

    df = pd.read_csv(io.StringIO(r.text), skiprows=1)
    df.columns = [c.lower().strip() for c in df.columns]
    if "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    else:
        raise ValueError("[X] No valid datetime column found")

    if "close" not in df.columns and "closeusd" in df.columns:
        df.rename(columns={"closeusd": "close"}, inplace=True)

    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"[✓] CDD loaded, rows={len(df)}, last={df['datetime'].max()}")
    return df


def fetch_binance(symbol: str, interval: str, start_time: datetime) -> pd.DataFrame:
    """从 Binance API 获取 start_time 之后的数据"""
    endpoint = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": 1000}
    params["startTime"] = int(start_time.timestamp() * 1000)
    try:
        r = requests.get(endpoint, params=params, timeout=10)
        if r.status_code != 200:
            print(f"[!] Binance fetch failed: HTTP {r.status_code}")
            return pd.DataFrame()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            print("[!] Binance returned empty data.")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "quote_asset_volume",
            "trades", "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.astype({
            "open": float, "high": float, "low": float,
            "close": float, "volume": float
        })
        print(f"[+] Binance 补齐数据 {len(df)} 行, 起点 {df['datetime'].min()}, 终点 {df['datetime'].max()}")
        return df
    except Exception as e:
        print(f"[!] Binance fetch error: {e}")
        return pd.DataFrame()


def merge_data(df_cdd: pd.DataFrame, df_binance: pd.DataFrame) -> pd.DataFrame:
    """合并并去重"""
    if df_binance.empty:
        print("[=] Binance 数据为空，保持原始 CDD 数据")
        return df_cdd

    merged = pd.concat([df_cdd, df_binance], ignore_index=True)
    merged = merged.drop_duplicates(subset=["datetime"], keep="last")
    merged = merged.sort_values("datetime").reset_index(drop=True)
    print(f"[✓] 合并后总行数={len(merged)}, 最新={merged['datetime'].max()}")
    return merged


def check_gaps(df: pd.DataFrame, freq: str):
    """检测时间连续性"""
    if freq not in ["1h", "4h", "1d"]:
        return
    df = df.sort_values("datetime")
    dt = df["datetime"].diff().dt.total_seconds().div(3600)
    gap_threshold = {"1h": 1.5, "4h": 5, "1d": 26}[freq]
    gaps = df.loc[dt > gap_threshold, "datetime"]
    if not gaps.empty:
        print(f"[!] 检测到 {len(gaps)} 个时间缺口：")
        for g in gaps:
            print(f"    - 缺口前时间点: {g - pd.Timedelta(hours=gap_threshold)} -> 缺口起点: {g}")
    else:
        print("[✓] 时间序列连续，无缺口")


# =========================
# 主执行入口
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--freq", required=True, choices=["1h", "4h", "1d"])
    parser.add_argument("--start", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    try:
        # Step 1️⃣ 下载 CDD
        df_cdd = fetch_cdd(args.symbol, args.freq)

        # Step 2️⃣ 计算增量补齐区间
        last_dt = df_cdd["datetime"].max()
        now_utc = datetime.now(timezone.utc)
        gap_hours = (now_utc - last_dt).total_seconds() / 3600

        if gap_hours > 1:
            interval = BINANCE_INTERVALS.get(args.freq, "1h")
            df_binance = fetch_binance(args.symbol, interval, last_dt)
            df_final = merge_data(df_cdd, df_binance)
        else:
            df_final = df_cdd
            print("[=] 数据已是最新，无需补齐")

        # Step 3️⃣ 检查时间连续性
        check_gaps(df_final, args.freq)

        # Step 4️⃣ 保存结果
        df_final.to_csv(args.out, index=False)
        print(f"[OK] Saved -> {args.out}  rows={len(df_final)}")

    except Exception as e:
        print(f"[X] 运行过程中出现错误: {e}")
        # 程序不中断，以便后续流程（如 auto_run.yml）继续
        try:
            if 'df_cdd' in locals():
                df_cdd.to_csv(args.out, index=False)
                print(f"[!] 已保存 CDD 原始数据 -> {args.out}")
        except Exception as ee:
            print(f"[X] 保存失败: {ee}")


if __name__ == "__main__":
    main()
