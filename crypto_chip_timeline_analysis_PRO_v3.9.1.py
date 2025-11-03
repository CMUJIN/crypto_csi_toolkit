#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crypto_chip_timeline_analysis_PRO_v4.0
---------------------------------------
✅ 保留 Chip Zone + Long/Short 全逻辑
✅ 增强 Liquidity 曲线绘制（独立时间范围）
✅ 文件命名包含 liquidity 起始日期
✅ 与 upload_to_notion.py 兼容
"""

import argparse, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ============ 基础工具 ============
def fmt_price(v: float) -> str:
    try:
        v = float(v)
        return f"{int(round(v))}" if v >= 100 else f"{v:.2f}"
    except Exception:
        return str(v)

def to_datetime_utc(series):
    return pd.to_datetime(series, errors="coerce", utc=True)

def ewma(series, span):
    return series.ewm(span=span, adjust=False, min_periods=max(span//2,1)).mean()

def zscore_rolling(series, window):
    mean = series.rolling(window, min_periods=max(window//10,1)).mean()
    std = series.rolling(window, min_periods=max(window//10,1)).std()
    return (series - mean) / std

# ============ 筹码分析 =============
def exp_half_life_decay(dt_series_utc, half_life_days: float):
    t = to_datetime_utc(dt_series_utc)
    tmax = t.max()
    if pd.isna(tmax):
        idx = np.arange(len(dt_series_utc))[::-1]
        age_days = idx / 24.0
    else:
        age_days = (tmax - t).dt.total_seconds().values / 86400.0
    lam = np.log(2.0) / max(half_life_days, 1e-9)
    return np.exp(-lam * age_days)

def compute_chipzones(df, window_zone=60, bins_pct=0.5, beta=0.7, half_life=10, quantile=0.8):
    for col in ["close", "volume", "datetime"]:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")
    close = df["close"].astype(float).values
    volume = df["volume"].astype(float).values

    if "open_interest" in df.columns:
        oi = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0)
        delta_oi = oi.diff().clip(lower=0).fillna(0).values
    else:
        delta_oi = np.zeros(len(df))

    decay = exp_half_life_decay(df["datetime"], half_life_days=float(half_life))
    weights = volume * np.power(1 + delta_oi, float(beta)) * decay

    mean_price = np.nanmean(close)
    bw = max(mean_price * (bins_pct/100), 1e-12)
    pmin, pmax = np.nanmin(close), np.nanmax(close)
    n_steps = int(np.ceil((pmax - pmin) / bw))
    bins = np.linspace(pmin, pmin + n_steps*bw, n_steps+1)
    lows, highs = bins[:-1], bins[1:]

    def hist_norm(prices, w, bins):
        h, _ = np.histogram(prices, bins=bins, weights=w)
        return (h / h.max()) if h.size and h.max()>0 else h

    h_all = hist_norm(close, weights, bins)
    h_recent = hist_norm(close[-window_zone:], weights[-window_zone:], bins)
    thr_all = np.quantile(h_all[h_all>0], quantile) if np.any(h_all>0) else 1.1
    thr_recent = np.quantile(h_recent[h_recent>0], quantile) if np.any(h_recent>0) else 1.1

    rows = []
    for i in range(len(lows)):
        ra, aa = h_recent[i], h_all[i]
        if (ra>=thr_recent) or (aa>=thr_all):
            zone_type = "persistent" if (ra>=thr_recent and aa>=thr_all) else "short-term"
            rows.append(dict(low=lows[i], high=highs[i], strength=(ra+aa)/2, zone_type=zone_type))
    return pd.DataFrame(rows).sort_values("strength", ascending=False)

# ============ Long/Short 分析 ============
def compute_long_short(close_series, eff_weight, window_strength, smooth, window_zone):
    close = close_series.astype(float)
    ret = close.pct_change().fillna(0)
    vol_scale = ret.rolling(window_zone, min_periods=1).std().replace(0, np.nan).fillna(ret.std() or 1)
    shifted_close = close.shift(1).bfill()
    impulse = (close.diff() / (vol_scale * shifted_close)) * (eff_weight / (np.nanmax(eff_weight) or 1))
    impulse = impulse.replace([np.inf, -np.inf], 0).fillna(0)
    impulse_s = impulse.rolling(window_strength, min_periods=1).mean()
    long_raw, short_raw = np.clip(impulse_s, 0, None), np.clip(-impulse_s, 0, None)
    norm = lambda x: (x / (x.max() or 1)).rolling(smooth, min_periods=1).mean().fillna(0)
    return norm(long_raw), norm(short_raw)

# ============ 流动性指标计算 ============
def detect_quote_volume_column(df):
    for c in df.columns:
        if ("quote" in c.lower() and "vol" in c.lower()) or ("usdt" in c.lower() and "volume" in c.lower()):
            return c
    return None

def compute_liquidity_series(df, span_hours=24, zwin_days=90):
    price = df["close"].astype(float)
    vol = df["volume"].astype(float)
    qv_col = detect_quote_volume_column(df)
    dollar_vol = pd.to_numeric(df[qv_col], errors="coerce") if qv_col else price * vol
    if "high" in df.columns and "low" in df.columns:
        high, low = pd.to_numeric(df["high"], errors="coerce"), pd.to_numeric(df["low"], errors="coerce")
        prev_close = price.shift(1)
        tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    else:
        tr = (price.pct_change().abs() * price).fillna(0)
    dv_ewm = ewma(dollar_vol, span_hours)
    tr_ewm = ewma(tr.replace(0, np.nan).fillna(tr.median()), span_hours)
    lp_hr = (dv_ewm / tr_ewm).replace([np.inf, -np.inf], np.nan)
    return zscore_rolling(lp_hr, int(24*zwin_days)), price

# ============ 流动性图绘制 ============
def draw_liquidity_curves(df_hourly, symbol, out_dir, span_hours, zwin_days, start_dt):
    df = df_hourly.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    warmup_start = (start_dt - pd.Timedelta(days=zwin_days+7)) if start_dt else None
    df_compute = df[df.index >= warmup_start] if warmup_start else df

    liq_z, price = compute_liquidity_series(df_compute.reset_index(), span_hours, zwin_days)
    liq_z.index, price.index = df_compute.index, df_compute.index
    if start_dt is not None:
        liq_z, price = liq_z[liq_z.index>=start_dt], price[price.index>=start_dt]

    liq_start_str = start_dt.strftime("%Y-%m-%d") if start_dt else "FULL"
    out_csv = out_dir / f"{symbol}_liquidity_{liq_start_str}.csv"
    out_png = out_dir / f"{symbol}_liquidity_{liq_start_str}.png"

    pd.DataFrame({"Liquidity_Z": liq_z, "Price": price}).dropna().to_csv(out_csv)
    print(f"[OK] Saved liquidity CSV → {out_csv}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.plot(price.index, price, color="tab:blue", label="Price (Left Axis)", linewidth=1.6)
    ax2.plot(liq_z.index, liq_z, color="tab:orange", label="Liquidity Score (Right Axis)", linewidth=1.6)

    # 标识线段
    mean, std = liq_z.mean(), liq_z.std()
    for lvl in [mean, mean+std, mean-std]:
        ax2.axhline(lvl, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USDT)")
    ax2.set_ylabel("Liquidity Score (Z-score)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title(f"{symbol} Liquidity Curve (from {liq_start_str})", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Saved liquidity chart → {out_png}")

# ============ 主程序入口 ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--start", required=False)
    ap.add_argument("--end", required=False)
    ap.add_argument("--liquidity_start", required=False)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.rename(columns=lambda x: x.strip().lower(), inplace=True)
    symbol = re.search(r"([A-Z]{3,5})USDT", args.csv.upper())
    symbol = symbol.group(1) + "USDT" if symbol else Path(args.csv).stem

    out_dir = Path("docs"); out_dir.mkdir(exist_ok=True)

    start_dt = pd.to_datetime(args.liquidity_start or args.start, errors="coerce", utc=True)
    print(f"[*] Liquidity start time = {start_dt}")

    draw_liquidity_curves(
        df_hourly=df.rename(columns={"date": "datetime"}),
        symbol=symbol,
        out_dir=out_dir,
        span_hours=24,
        zwin_days=90,
        start_dt=start_dt
    )

if __name__ == "__main__":
    main()
