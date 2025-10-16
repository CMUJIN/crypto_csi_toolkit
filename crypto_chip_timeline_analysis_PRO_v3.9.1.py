#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Chip Strength Timeline Analysis (v3.9.2)
------------------------------------------------
Usage:
  python crypto_chip_timeline_analysis_PRO_v3.9.2.py <CSV_FILE> \
    --window_strength 5 --window_zone 60 --bins_pct 0.5 \
    --beta 0.7 --half_life 10000000 --quantile 0.8 \
    --ma_period 10 --smooth 3
"""

import argparse, sys, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------
def rolling_zscore(series, window):
    """Compute rolling z-score for momentum indication."""
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-9)


# ---------------------------------------------------
def compute_strength(df, window_strength=5, window_zone=60, bins_pct=0.5,
                     beta=0.7, half_life=10000000, quantile=0.8):
    """Compute chip strength distribution by volume density over price."""
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    n = len(df)
    bins = int((close.max() - close.min()) / (close.mean() * bins_pct / 100)) + 1
    bins = max(bins, 10)
    ranges = np.linspace(close.min(), close.max(), bins + 1)

    strength_map = np.zeros(bins)
    decay = np.exp(-np.arange(window_zone)[::-1] / half_life)

    for i in range(window_zone, n):
        sub_close = close[i - window_zone:i]
        sub_vol = volume[i - window_zone:i]

        hist, _ = np.histogram(sub_close, bins=ranges, weights=sub_vol)
        hist = hist / (hist.sum() + 1e-9)

        # ✅ 修复维度错误：使用成交量平均衰减权重代替 dot()
        w = np.average(hist) * beta + (1 - beta) * np.mean(decay)
        strength_map += hist * w

    if strength_map.max() > 0:
        strength_map /= strength_map.max()

    return pd.DataFrame({
        "low": ranges[:-1],
        "high": ranges[1:],
        "strength": strength_map
    })


# ---------------------------------------------------
def draw_timeline(df, bins_df, out_png, ma_period=10, quantile=0.8, smooth=3):
    """Draw price, volume, and chip strength timeline chart."""
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    t = df["datetime"]
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    ma_vol = volume.rolling(ma_period).mean()

    fig, (axa, axb) = plt.subplots(2, 1, figsize=(10, 6),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # --- Price line ---
    axa.plot(t, close, color="black", linewidth=0.9, label="price")

    # --- Long/short momentum ---
    long_strength = rolling_zscore(close, smooth).clip(lower=0)
    short_strength = (-rolling_zscore(close, smooth)).clip(lower=0)
    axa.plot(t, long_strength, color="red", linewidth=0.8, label="long")
    axa.plot(t, short_strength, color="green", linewidth=0.8, label="short")

    axa.legend(loc="upper right")
    axa.set_ylabel("price")

    # --- Draw chip strength zones ---
    for _, row in bins_df.iterrows():
        alpha_val = 0.15 + 0.35 * row["strength"]
        axa.axhspan(row["low"], row["high"], color="orange", alpha=alpha_val)
        # Randomly annotate a few zones
        if np.random.rand() < 0.05:
            val_low = row["low"]
            val_high = row["high"]
            label_txt = (f"{val_low:.0f}-{val_high:.0f}"
                         if val_high > 100 else f"{val_low:.2f}-{val_high:.2f}")
            axa.text(val_low, val_high,
                     f"{label_txt}\nstrength:{row['strength']:.2f}",
                     fontsize=6, color="brown", alpha=0.8)

    # --- Volume subplot ---
    axb.bar(t, volume, color="gray", alpha=0.6)
    axb.plot(t, ma_vol, color="blue", linewidth=0.8, label="vol MA")
    axb.legend(loc="upper right")
    axb.set_ylabel("volume")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[OK] Saved timeline figure -> {out_png}")


# ---------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Input CSV file")
    ap.add_argument("--window_strength", type=int, default=5)
    ap.add_argument("--window_zone", type=int, default=60)
    ap.add_argument("--bins_pct", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--half_life", type=float, default=10000000)
    ap.add_argument("--quantile", type=float, default=0.8)
    ap.add_argument("--ma_period", type=int, default=10)
    ap.add_argument("--smooth", type=int, default=3)
    args = ap.parse_args()

    # --- Load CSV ---
    df = pd.read_csv(args.csv)

    # --- ✅ Normalize columns (case-insensitive) ---
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {}
    for target, cands in {
        "open": ["open", "opening"],
        "high": ["high"],
        "low": ["low"],
        "close": ["close", "closing", "last"],
        "volume": ["volume", "volume usdt", "volumeusd", "volume usdt.1", "vol"],
        "datetime": ["datetime", "date", "timestamp"]
    }.items():
        for c in cands:
            if c in df.columns:
                rename_map[c] = target
                break
    df = df.rename(columns=rename_map)

    # --- ✅ Auto-filter start date from filename ---
    m = re.search(r'_(\d{4}-\d{2}-\d{2})_to_', args.csv)
    if m:
        start_dt = pd.to_datetime(m.group(1), utc=True)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df[df["datetime"] >= start_dt]
        print(f"[*] Filtered data from {start_dt.strftime('%Y-%m-%d')} onward, rows={len(df)}")

    # --- Validate required columns ---
    for col in ["close", "volume", "datetime"]:
        if col not in df.columns:
            print(f"[X] Missing required column: {col}. Columns = {list(df.columns)}")
            sys.exit(1)

    # --- Compute chip strength ---
    bins_df = compute_strength(df, args.window_strength, args.window_zone,
                               args.bins_pct, args.beta, args.half_life, args.quantile)

    out_csv = args.csv.replace(".csv", "_chip_strength.csv")
    bins_df.to_csv(out_csv, index=False)
    print(f"[OK] Saved chip_strength table -> {out_csv}")

    out_png = args.csv.replace(".csv", "_chip_timeline_pro.png")
    draw_timeline(df, bins_df, out_png,
                  ma_period=args.ma_period,
                  quantile=args.quantile,
                  smooth=args.smooth)


if __name__ == "__main__":
    main()
