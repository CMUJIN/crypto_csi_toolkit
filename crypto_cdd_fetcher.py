#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Crypto Data Fetcher (v2.4-stable)
-------------------------------------------
✓ 下载 CDD 历史数据
✓ 严格从 --start 日期起输出（早于 start 的数据不会写入）
✓ 自动尝试用 Binance 实时接口补齐（到当前小时/日）
   - 先用 https://api.binance.com
   - 若 451/网络异常，再用 https://data-api.binance.vision
✓ Binance 接口失败不影响主流程（打印日志后继续）
✓ 检测时间缺口（基于过滤后的区间）
"""

import argparse, os, io, requests, pandas as pd
from datetime import datetime, timezone

BINANCE_INTERVALS = {"1h": "1h", "4h": "4h", "1d": "1d"}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CryptoCSI/2.4)"}

def fetch_cdd(symbol: str, freq: str) -> pd.DataFrame:
    """从 CryptoDataDownload 获取基础数据"""
    url = f"https://www.cryptodatadownload.com/cdd/Binance_{symbol}_{freq}.csv"
    print(f"[*] Fetching: {url}")
    r = requests.get(url, timeout=20, headers=HEADERS)
    if r.status_code != 200:
        raise ValueError(f"[X] Download failed: HTTP {r.status_code}")

    # CDD 第一行多为注释，跳过
    df = pd.read_csv(io.StringIO(r.text), skiprows=1)
    df.columns = [c.lower().strip() for c in df.columns]

    # 兼容 datetime 列
    if "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    else:
        raise ValueError("[X] No valid datetime column found in CDD CSV")

    # 兼容收盘价字段
    if "close" not in df.columns and "closeusd" in df.columns:
        df.rename(columns={"closeusd": "close"}, inplace=True)

    # 兼容成交量字段（尽量保留原列名，不强制重命名）
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    print(f"[✓] CDD loaded, rows={len(df)}, last={df['datetime'].max()}")
    return df


def _binance_endpoint_list():
    # 主端点 + 公开镜像端点
    return [
        "https://api.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
    ]


def fetch_binance(symbol: str, interval: str, start_time: datetime) -> pd.DataFrame:
    """从 Binance 获取 start_time 之后的数据；失败返回空 DataFrame"""
    params = {"symbol": symbol, "interval": interval, "limit": 1000}
    params["startTime"] = int(start_time.timestamp() * 1000)

    for ep in _binance_endpoint_list():
        try:
            r = requests.get(ep, params=params, timeout=12, headers=HEADERS)
            if r.status_code == 451:
                print(f"[!] Binance 451 restricted at {ep}; trying fallback...")
                continue
            if r.status_code != 200:
                print(f"[!] Binance fetch failed: HTTP {r.status_code} @ {ep}")
                continue

            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                print(f"[!] Binance returned empty list @ {ep}")
                continue

            df = pd.DataFrame(
                data,
                columns=[
                    "open_time", "open", "high", "low", "close",
                    "volume", "close_time", "quote_asset_volume",
                    "trades", "taker_buy_base", "taker_buy_quote", "ignore"
                ],
            )
            df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df = df.astype({
                "open": float, "high": float, "low": float,
                "close": float, "volume": float
            })
            df = df[["datetime", "open", "high", "low", "close", "volume"]].sort_values("datetime")
            print(f"[+] Binance 补齐数据 {len(df)} 行, 起点 {df['datetime'].min()}, 终点 {df['datetime'].max()} via {ep}")
            return df
        except Exception as e:
            print(f"[!] Binance fetch error @ {ep}: {e}")

    # 全部失败
    return pd.DataFrame()


def merge_data(df_cdd: pd.DataFrame, df_binance: pd.DataFrame) -> pd.DataFrame:
    """合并并去重（datetime 唯一）"""
    if df_binance.empty:
        print("[=] Binance 数据为空，保持原始 CDD 数据")
        return df_cdd

    merged = pd.concat([df_cdd, df_binance], ignore_index=True)
    merged = merged.drop_duplicates(subset=["datetime"], keep="last")
    merged = merged.sort_values("datetime").reset_index(drop=True)
    print(f"[✓] 合并后总行数={len(merged)}, 最新={merged['datetime'].max()}")
    return merged


def check_gaps(df: pd.DataFrame, freq: str):
    """检测时间连续性（基于 freq）"""
    if freq not in ["1h", "4h", "1d"] or df.empty:
        return
    df = df.sort_values("datetime")
    dt_hours = df["datetime"].diff().dt.total_seconds().div(3600)
    gap_threshold = {"1h": 1.5, "4h": 5, "1d": 26}[freq]
    gaps = df.loc[dt_hours > gap_threshold, "datetime"]
    if not gaps.empty:
        print(f"[!] 检测到 {len(gaps)} 个时间缺口（已过滤后的区间）：")
        for g in gaps.head(10):  # 避免日志过长，仅前10个
            print(f"    - 缺口起点: {g}")
        if len(gaps) > 10:
            print(f"    ... 其余 {len(gaps)-10} 个省略")
    else:
        print("[✓] 时间序列连续，无明显缺口")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--freq", required=True, choices=["1h", "4h", "1d"])
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 解析 start（统一为 UTC）
    try:
        start_dt = pd.to_datetime(args.start, utc=True)
    except Exception:
        raise ValueError(f"[X] Invalid --start: {args.start}")

    try:
        # 1) CDD 历史
        df_cdd = fetch_cdd(args.symbol, args.freq)

        # 2) 增量补齐（以 CDD 最新时间为起点）
        last_dt = df_cdd["datetime"].max()
        now_utc = datetime.now(timezone.utc)
        gap_hours = (now_utc - last_dt).total_seconds() / 3600

        if gap_hours > 1:
            interval = BINANCE_INTERVALS.get(args.freq, "1h")
            df_binance = fetch_binance(args.symbol, interval, last_dt)
            df_all = merge_data(df_cdd, df_binance)
        else:
            print("[=] CDD 已接近最新，无需补齐")
            df_all = df_cdd

        # 3) 关键修正：**仅保留 >= start 的数据**
        df_final = df_all[df_all["datetime"] >= start_dt].copy()
        df_final = df_final.sort_values("datetime").reset_index(drop=True)
        print(f"[*] Filtered to start >= {start_dt.date()}, rows={len(df_final)} | "
              f"first={df_final['datetime'].min() if not df_final.empty else 'NA'} | "
              f"last={df_final['datetime'].max() if not df_final.empty else 'NA'}")

        # 4) 基于过滤后的数据做连续性检查
        check_gaps(df_final, args.freq)

        # 5) 保存
        df_final.to_csv(args.out, index=False)
        print(f"[OK] Saved -> {args.out}  rows={len(df_final)}")

    except Exception as e:
        print(f"[X] 运行过程中出现错误: {e}")
        # 兜底：尽可能保存 >= start 的 CDD 子集
        try:
            df_cdd = df_cdd[df_cdd["datetime"] >= start_dt].sort_values("datetime")
            df_cdd.to_csv(args.out, index=False)
            print(f"[!] 已保存 CDD（>=start）数据 -> {args.out} rows={len(df_cdd)}")
        except Exception as ee:
            print(f"[X] 兜底保存失败: {ee}")


if __name__ == "__main__":
    main()
