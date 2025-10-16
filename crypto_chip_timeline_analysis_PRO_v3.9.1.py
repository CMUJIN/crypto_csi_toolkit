#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Chip Timeline Analysis (v3.9.5)
--------------------------------------
✓ 修复 tz-naive vs tz-aware datetime 报错
✓ 保留筹码密集区过滤逻辑 (quantile)
✓ 恢复多空吸筹曲线
✓ 恢复 strength 区间标注
✓ 自动纵坐标范围 (不从0开始)
"""

import argparse, sys, re, os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ==============================
# Utility Functions
# ==============================
def fmt_price(v: float) -> str:
    """格式化价格显示（整数或两位小数）"""
    return f"{int(round(v))}" if v >= 100 else f"{v:.2f}"


def rolling_zscore(series, window):
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-9)


def exp_decay_weights(length: int, half_life: float):
    idx = np.arange(length, dtype=float)
    age = (length - 1) - idx
    lam = np.log(2.0) / max(half_life, 1e-9)
    return np.exp(-lam * age)


def compute_effective_volume(vol: np.ndarray, beta: float, half_life: float):
    L = len(vol)
    dec = exp_decay_weights(L, half_life)
    mix = beta + (1.0 - beta) * dec
    return vol * mix


# ==============================
# Core Computation
# ==============================
def compute_bins(prices, eff_vol, bin_pct):
    pmin, pmax = float(np.nanmin(prices)), float(np.nanmax(prices))
    width = bin_pct / 100.0
    n_bins = int(np.clip(np.ceil((pmax - pmin) / (pmin * width)), 30, 600))
    bins = np.linspace(pmin, pmax, n_bins + 1)
    idx = np.clip(np.digitize(prices, bins) - 1, 0, n_bins - 1)
    vol_sum = np.bincount(idx, weights=eff_vol, minlength=n_bins).astype(float)
    vol_sum /= (vol_sum.max() or 1.0)
    return pd.DataFrame({"low": bins[:-1], "high": bins[1:], "strength": vol_sum})


def compute_long_short(close, eff_vol, window_strength, smooth, window_zone):
    close = close.astype(float)
    ret = close.pct_change().fillna(0)
    vol_scale = ret.rolling(window_zone, min_periods=1).std().replace(0.0, np.nan).fillna(ret.std() or 1.0)
    shifted_close = close.shift(1).bfill().replace(0, np.nan).bfill()
    impulse = (close.diff().fillna(0.0) / (vol_scale * shifted_close)) * (eff_vol / (eff_vol.max() or 1.0))
    impulse = impulse.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    impulse_s = impulse.rolling(window_strength, min_periods=1).mean()
    long_raw = np.clip(impulse_s, 0, None)
    short_raw = np.clip(-impulse_s, 0, None)
    norm = lambda x: (x / (x.max() or 1.0)).rolling(smooth, min_periods=1).mean().fillna(0.0)
    return norm(long_raw), norm(short_raw)


# ==============================
# Visualization
# ==============================
def draw_timeline(df, bins_df, out_png, ma_period, quantile, long_curve, short_curve):
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    t = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    price = df["close"].astype(float)
    volume = df["volume"].astype(float)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.4], hspace=0.15)
    ax = fig.add_subplot(gs[0, 0]); ax2 = ax.twinx()

    # --- Price line ---
    ax.plot(t, price, color="black", linewidth=1.2, label="price")
    ax.set_ylabel("price")
    ax2.set_ylabel("strength")

    # --- Chip zones filtering by quantile ---
    threshold = bins_df["strength"].quantile(quantile)
    used_y = []
    for _, r in bins_df.iterrows():
        if r["strength"] >= threshold:
            ax.axhspan(r["low"], r["high"], color=(1.0, 0.65, 0.0, 0.25))
            y_mid = (r["low"] + r["high"]) / 2.0
            label = f"{fmt_price(r['low'])}-{fmt_price(r['high'])}  strength:{r['strength']:.2f}"
            if not any(abs(y_mid - yy) < (price.max() - price.min()) * 0.01 for yy in used_y):
                ax.text(t.iloc[0], y_mid, label, fontsize=8, color="saddlebrown",
                        bbox=dict(facecolor=(1, 0.9, 0.8, 0.7),
                                  edgecolor="none", boxstyle="round,pad=0.2"))
                used_y.append(y_mid)

    # --- Long/Short curves ---
    ax2.plot(t, long_curve, color="red", linewidth=1.1, label="long")
    ax2.plot(t, short_curve, color="green", linewidth=1.1, label="short")
    ax2.set_ylim(0, 1.0)

    # --- Legend ---
    ln1, = ax.plot([], [], color="black", label="price")
    ln2, = ax2.plot([], [], color="red", label="long")
    ln3, = ax2.plot([], [], color="green", label="short")
    ax.legend(handles=[ln1, ln2, ln3], loc="upper right", frameon=True)

    # --- Volume subplot ---
    axb = fig.add_subplot(gs[1, 0], sharex=ax)
    axb.bar(t, volume, color="gray", alpha=0.7)
    vol_ma = volume.rolling(window=ma_period, min_periods=1).mean()
    axb.plot(t, vol_ma, color="blue", linewidth=1.2, label="vol MA")
    axb.legend(loc="upper right")
    axb.set_ylabel("volume")

    # --- Dynamic price range ---
    ymin, ymax = price.min(), price.max()
    ax.set_ylim(ymin * 0.98, ymax * 1.02)
    for label in axb.get_xticklabels():
        label.set_rotation(30); label.set_ha("right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved timeline figure -> {out_png}")


# ==============================
# Main
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV file")
    ap.add_argument("--window_strength", type=int, default=5)
    ap.add_argument("--window_zone", type=int, default=60)
    ap.add_argument("--bins_pct", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--half_life", type=float, default=10000000)
    ap.add_argument("--quantile", type=float, default=0.8)
    ap.add_argument("--ma_period", type=int, default=10)
    ap.add_argument("--smooth", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.lower().strip() for c in df.columns]
    rename_map = {}
    for target, cands in {
        "open": ["open"], "high": ["high"], "low": ["low"],
        "close": ["close", "last"], "volume": ["volume", "volumeusd", "volume usdt"],
        "datetime": ["datetime", "date", "timestamp"]
    }.items():
        for c in cands:
            if c in df.columns:
                rename_map[c] = target; break
    df = df.rename(columns=rename_map)

    # --- Auto date filter from filename (tz-safe) ---
    m = re.search(r'_(\d{4}-\d{2}-\d{2})_to_', args.csv)
    if m:
        start_dt = pd.to_datetime(m.group(1), utc=True)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df[df["datetime"] >= start_dt]
        print(f"[*] Filtered data from {start_dt.strftime('%Y-%m-%d')} onward, rows={len(df)}")

    for col in ["close", "volume", "datetime"]:
        if col not in df.columns:
            sys.exit(f"[X] Missing column: {col}")

    eff_vol = compute_effective_volume(df["volume"].values, beta=args.beta, half_life=args.half_life)
    bins_df = compute_bins(df["close"].values, eff_vol, args.bins_pct)
    long_curve, short_curve = compute_long_short(df["close"], eff_vol,
                                                 args.window_strength, args.smooth, args.window_zone)

    out_csv = args.csv.replace(".csv", "_chip_strength.csv")
    bins_df.to_csv(out_csv, index=False)
    print(f"[OK] Saved chip_strength table -> {out_csv}")

    out_png = args.csv.replace(".csv", "_chip_timeline_pro.png")
    draw_timeline(df, bins_df, out_png, args.ma_period, args.quantile,
                  long_curve, short_curve)


if __name__ == "__main__":
    main()
