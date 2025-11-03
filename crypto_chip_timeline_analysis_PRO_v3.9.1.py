#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Chip Timeline Analysis (v4.0 hybrid fixed)
-------------------------------------------------
✅ 保留原筹码分析逻辑
✅ 增加流动性曲线绘制 (Liquidity Curve)
✅ 输出互不干扰：
    - *_chip_strength.csv
    - *_chip_timeline_pro.png
    - *_liquidity_<start>.png
"""

import argparse, sys, re, os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

# ======================================
# Utilities
# ======================================
def fmt_price(v: float) -> str:
    try:
        v = float(v)
        return f"{int(round(v))}" if v >= 100 else f"{v:.2f}"
    except Exception:
        return str(v)


def to_utc_datetime(s):
    return pd.to_datetime(s, errors="coerce", utc=True)


def exp_half_life_decay(dt_series_utc: pd.Series, half_life_days: float) -> np.ndarray:
    t = to_utc_datetime(dt_series_utc)
    t_max = t.max()
    if pd.isna(t_max):
        idx = np.arange(len(dt_series_utc))[::-1]
        age_days = idx / 24.0
    else:
        age_days = (t_max - t).dt.total_seconds().values / 86400.0

    half_life_days = max(float(half_life_days), 1e-9)
    lam = np.log(2.0) / half_life_days
    return np.exp(-lam * age_days)


# ======================================
# Chip Zones (Futures Logic)
# ======================================
def compute_chipzones_futures_logic(
    df: pd.DataFrame,
    window_zone: int = 60,
    bins_pct: float = 0.5,
    beta: float = 0.7,
    half_life: float = 10.0,
    quantile: float = 0.8,
):
    for col in ["close", "volume", "datetime"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    close = df["close"].astype(float).values
    volume = df["volume"].astype(float).values
    dt = df["datetime"]

    if "open_interest" in df.columns:
        oi = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0.0)
        delta_oi = oi.diff().clip(lower=0.0).fillna(0.0).values
    else:
        delta_oi = np.zeros(len(df))

    decay = exp_half_life_decay(df["datetime"], half_life_days=float(half_life))
    weights = volume * np.power(1.0 + delta_oi, float(beta)) * decay

    mean_price = np.nanmean(close)
    bw = max(mean_price * (float(bins_pct) / 100.0), 1e-12)
    pmin, pmax = float(np.nanmin(close)), float(np.nanmax(close))
    n_steps = max(int(np.ceil((pmax - pmin) / bw)), 1)
    bins = np.linspace(pmin, pmin + n_steps * bw, n_steps + 1)
    lows, highs = bins[:-1], bins[1:]

    def hist_norm(prices, w, _bins):
        h, _ = np.histogram(prices, bins=_bins, weights=w)
        mx = h.max() if h.size else 0.0
        return (h / mx) if mx > 0 else h

    h_all = hist_norm(close, weights, bins)
    win = max(int(window_zone), 1)
    h_recent = hist_norm(close[-win:], weights[-win:], bins)

    thr_all = np.quantile(h_all[h_all > 0], quantile) if np.any(h_all > 0) else 1.1
    thr_recent = np.quantile(h_recent[h_recent > 0], quantile) if np.any(h_recent > 0) else 1.1

    rows = []
    for i in range(len(lows)):
        ra, aa = float(h_recent[i]), float(h_all[i])
        if (ra >= thr_recent) or (aa >= thr_all):
            avg = (ra + aa) / 2.0
            persistent = (ra >= thr_recent) and (aa >= thr_all)
            rows.append(dict(
                low=lows[i], high=highs[i],
                recent_strength=ra, all_strength=aa,
                avg_strength=avg, zone_type="persistent" if persistent else "short-term",
                strength=avg,
            ))

    if not rows:
        avg_all = (h_recent + h_all) / 2.0
        order = np.argsort(-avg_all)
        for j in order[:10]:
            rows.append(dict(
                low=lows[j], high=highs[j],
                recent_strength=float(h_recent[j]),
                all_strength=float(h_all[j]),
                avg_strength=float(avg_all[j]),
                zone_type="short-term", strength=float(avg_all[j]),
            ))

    chip_df = pd.DataFrame(rows)
    return chip_df.sort_values("avg_strength", ascending=False).reset_index(drop=True)


# ======================================
# Long / Short Curves
# ======================================
def compute_long_short(close_series: pd.Series, eff_weight: np.ndarray,
                       window_strength: int, smooth: int, window_zone: int):
    close = close_series.astype(float)
    ret = close.pct_change().fillna(0)
    vol_scale = ret.rolling(window_zone, min_periods=1).std().replace(0, np.nan)
    vol_scale = vol_scale.fillna(ret.std() or 1.0)

    shifted_close = close.shift(1).bfill().replace(0, np.nan).bfill()
    impulse = (close.diff().fillna(0.0) / (vol_scale * shifted_close)) * (
        eff_weight / (np.nanmax(eff_weight) or 1.0)
    )
    impulse = impulse.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    impulse_s = impulse.rolling(window_strength, min_periods=1).mean()
    long_raw = np.clip(impulse_s, 0, None)
    short_raw = np.clip(-impulse_s, 0, None)

    norm = lambda x: (x / (x.max() or 1.0)).rolling(smooth, min_periods=1).mean().fillna(0.0)
    return norm(long_raw), norm(short_raw)


# ======================================
# Visualization
# ======================================
def draw_timeline(df, chip_df, out_png, ma_period, long_curve, short_curve):
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    t = to_utc_datetime(df["datetime"])
    price = df["close"].astype(float)
    volume = df["volume"].astype(float)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.4], hspace=0.15)
    ax = fig.add_subplot(gs[0, 0]); ax2 = ax.twinx()

    ax.plot(t, price, color="black", linewidth=1.2, label="price")
    ax2.set_ylabel("strength")

    for _, r in chip_df.iterrows():
        color = (1, 0.55, 0.0, 0.30) if r["zone_type"] == "persistent" else (1, 0.8, 0.2, 0.20)
        ax.axhspan(r["low"], r["high"], color=color)

    ax2.plot(t, long_curve, color="red", linewidth=1.0, label="long")
    ax2.plot(t, short_curve, color="green", linewidth=1.0, label="short")
    ax.legend(loc="upper right")

    axb = fig.add_subplot(gs[1, 0], sharex=ax)
    axb.bar(t, volume, color="gray", alpha=0.7)
    axb.plot(t, volume.rolling(ma_period, min_periods=1).mean(), color="blue", linewidth=1.2)

    for label in axb.get_xticklabels():
        label.set_rotation(30); label.set_ha("right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved timeline figure -> {out_png}")


# ======================================
# Liquidity Curve
# ======================================
def draw_liquidity_curve(df, liquidity_start, symbol):
    df = df.copy()
    df["datetime"] = to_utc_datetime(df["datetime"])
    df = df[df["datetime"] >= pd.to_datetime(liquidity_start, utc=True)]
    df = df.set_index("datetime")

    if df.empty:
        print(f"[!] No data available for liquidity curve since {liquidity_start}")
        return

    price = df["close"].astype(float)
    liq_score = (df["volume"].astype(float).rolling(24).mean() /
                 price.rolling(24).std().replace(0, np.nan))

    liq_score = (liq_score - liq_score.mean()) / (liq_score.std() or 1)
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    ax1.plot(df.index, price, color="black", label="Price")
    ax2.plot(df.index, liq_score, color="blue", label="Liquidity Score")
    ax1.set_ylabel("Price")
    ax2.set_ylabel("Liquidity Z-score")

    for y in [-1, 0, 1]:
        ax2.axhline(y, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle(f"{symbol} Liquidity Curve since {liquidity_start}", fontsize=12)
    fig.tight_layout()
    out_png = f"docs/{symbol}_liquidity_{liquidity_start}.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved liquidity curve -> {out_png}")


# ======================================
# Main
# ======================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--window_strength", type=int, default=5)
    ap.add_argument("--window_zone", type=int, default=60)
    ap.add_argument("--bins_pct", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--half_life", type=float, default=10)
    ap.add_argument("--quantile", type=float, default=0.8)
    ap.add_argument("--ma_period", type=int, default=10)
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--liquidity_start", type=str, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.lower().strip() for c in df.columns]
    rename_map = {}
    col_map = {
        "open": ["open"], "high": ["high"], "low": ["low"],
        "close": ["close", "last"], "volume": ["volume", "volumeusd", "volume usdt"],
        "datetime": ["datetime", "date", "timestamp"]
    }
    for target, cands in col_map.items():
        for c in cands:
            if c in df.columns: rename_map[c] = target; break
    df = df.rename(columns=rename_map)
    df["datetime"] = to_utc_datetime(df["datetime"])

    # Filter start
    m = re.search(r'_(\d{4}-\d{2}-\d{2})_to_', args.csv)
    if m:
        start_dt = pd.to_datetime(m.group(1), utc=True)
        df = df[df["datetime"] >= start_dt]
        print(f"[*] Filtered from {start_dt.date()}, rows={len(df)}")

    for col in ["close", "volume"]:
        if col not in df.columns:
            raise SystemExit(f"[X] Missing column: {col}")

    chip_df = compute_chipzones_futures_logic(df, args.window_zone, args.bins_pct,
                                              args.beta, args.half_life, args.quantile)

    if "open_interest" in df.columns:
        oi = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0.0)
        delta_oi = oi.diff().clip(lower=0.0).fillna(0.0).values
    else:
        delta_oi = np.zeros(len(df))
    decay = exp_half_life_decay(df["datetime"], args.half_life)
    eff_weight = df["volume"].astype(float).values * np.power(1.0 + delta_oi, float(args.beta)) * decay

    long_curve, short_curve = compute_long_short(df["close"], eff_weight,
                                                 args.window_strength, args.smooth, args.window_zone)

    symbol = os.path.basename(args.csv).split("_")[1]
    os.makedirs("docs", exist_ok=True)
    chip_csv = f"docs/{symbol}_chip_strength.csv"
    chip_df.to_csv(chip_csv, index=False)
    print(f"[OK] Saved chip_strength table -> {chip_csv}")

    chip_png = f"docs/{symbol}_chip_timeline_pro.png"
    draw_timeline(df, chip_df, chip_png, args.ma_period, long_curve, short_curve)

    if args.liquidity_start:
        draw_liquidity_curve(df, args.liquidity_start, symbol)


if __name__ == "__main__":
    main()
