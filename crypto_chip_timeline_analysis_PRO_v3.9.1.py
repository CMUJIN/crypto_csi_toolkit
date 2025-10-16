
# -*- coding: utf-8 -*-
"""
Crypto Timeline Chip Analysis (PRO v3.9.1)
------------------------------------------
Fixes:
- Remove FutureWarning (no deprecated fillna(method='bfill'))
- Normalize CSI (strength) to 0–1 range
- Preserve volume_sum column in CSV
"""

import os, sys, argparse, numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def find_col(df, candidates):
    for name in candidates:
        for col in df.columns:
            if col.lower().strip().replace(' ', '') == name.lower().strip().replace(' ', ''):
                return col
    return None

def fmt_price(v: float) -> str:
    return f"{int(round(v))}" if v >= 100 else f"{v:.2f}"

def exp_decay_weights(length: int, half_life: float):
    if half_life <= 0:
        return np.ones(length, dtype=float)
    idx = np.arange(length, dtype=float)
    age = (length - 1) - idx
    lam = np.log(2.0) / max(half_life, 1e-9)
    return np.exp(-lam * age)

def compute_effective_volume(vol: np.ndarray, beta: float, half_life: float):
    L = len(vol)
    dec = exp_decay_weights(L, half_life)
    mix = beta + (1.0 - beta) * dec
    return vol * mix

def compute_bins(prices: np.ndarray, eff_vol: np.ndarray, bin_pct: float):
    pmin, pmax = float(np.nanmin(prices)), float(np.nanmax(prices))
    if pmin <= 0: pmin = max(1e-6, pmin)
    width = bin_pct / 100.0
    n_bins = int(np.clip(np.ceil((pmax - pmin) / (pmin * width)), 30, 600))
    bins = np.linspace(pmin, pmax, n_bins + 1)
    idx = np.clip(np.digitize(prices, bins) - 1, 0, n_bins - 1)
    vol_sum = np.bincount(idx, weights=eff_vol, minlength=n_bins).astype(float)
    total_vol = vol_sum.sum() or 1.0
    # Normalize CSI 0–1
    csi = vol_sum / (vol_sum.max() or 1.0)
    df_out = pd.DataFrame({
        "price_bin_low": bins[:-1],
        "price_bin_high": bins[1:],
        "bin_mid": (bins[:-1] + bins[1:]) / 2.0,
        "volume_sum": vol_sum,
        "csi": csi,
        "pct_of_total_volume": vol_sum / total_vol * 100.0,
    }).sort_values("price_bin_low").reset_index(drop=True)
    for c in ["price_bin_low","price_bin_high","bin_mid","volume_sum","csi","pct_of_total_volume"]:
        df_out[c] = df_out[c].astype(float).round(4)
    return df_out

def compute_long_short(close: pd.Series, eff_vol: np.ndarray, window_strength: int, smooth: int, window_zone: int):
    close = close.astype(float)
    ret = close.pct_change().fillna(0.0)
    vol_scale = ret.rolling(window=window_zone, min_periods=1).std().replace(0.0, np.nan).fillna(ret.std() or 1.0)
    shifted_close = close.shift(1).bfill().replace(0, np.nan).bfill()
    impulse = (close.diff().fillna(0.0) / (vol_scale * shifted_close)) * (eff_vol / (eff_vol.max() or 1.0))
    impulse = impulse.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    impulse_s = impulse.rolling(window=window_strength, min_periods=1).mean()
    long_raw = np.clip(impulse_s, 0, None)
    short_raw = np.clip(-impulse_s, 0, None)
    def norm_s(x):
        m = x.max()
        y = x / m if m and m > 0 else x * 0
        if smooth and smooth > 1:
            y = y.rolling(window=smooth, min_periods=1).mean()
        return y.fillna(0.0)
    return norm_s(long_raw), norm_s(short_raw)

def draw_timeline(df, bins_df, out_png, ma_period, quantile, long_curve, short_curve):
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    t = pd.to_datetime(df["datetime"])
    price = df["close"].astype(float)
    volume = df["volume"].astype(float)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.4], hspace=0.15)
    ax = fig.add_subplot(gs[0, 0]); ax2 = ax.twinx()

    ax.plot(t, price, color="black", linewidth=1.4, label="price")
    ax.set_ylabel("price"); ax2.set_ylabel("strength")

    threshold = bins_df["csi"].quantile(quantile)
    for _, r in bins_df.iterrows():
        if r["csi"] >= threshold and r["csi"] > 0:
            ax.axhspan(float(r["price_bin_low"]), float(r["price_bin_high"]), color=(1.0, 0.65, 0.0, 0.25))
    x_min, x_max = ax.get_xlim()
    x_text = x_min + 0.01 * (x_max - x_min)
    used_y = []
    for _, r in bins_df.iterrows():
        if r["csi"] >= threshold and r["csi"] > 0:
            y0, y1 = float(r["price_bin_low"]), float(r["price_bin_high"])
            label = f"{fmt_price(y0)}–{fmt_price(y1)}  strength:{float(r['csi']):.2f}"
            y_mid = (y0 + y1) / 2.0
            dy = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012
            while any(abs(y_mid - yy) < dy for yy in used_y):
                y_mid += dy
            used_y.append(y_mid)
            ax.text(t.iloc[0], y_mid, label, va="center", ha="left",
                    bbox=dict(facecolor=(1, 0.9, 0.8, 0.8), edgecolor="none", boxstyle="round,pad=0.2"),
                    fontsize=9, color="saddlebrown")

    ax2.plot(t, long_curve.values, color="red", linewidth=1.2, label="long")
    ax2.plot(t, short_curve.values, color="green", linewidth=1.2, label="short")
    ax2.set_ylim(0, 1.0)

    ln1, = ax.plot([], [], color="black", label="price")
    ln2, = ax2.plot([], [], color="red", label="long")
    ln3, = ax2.plot([], [], color="green", label="short")
    ax.legend(handles=[ln1, ln2, ln3], loc="upper right", frameon=True)

    axb = fig.add_subplot(gs[1, 0], sharex=ax)
    t_num = mdates.date2num(t)
    bar_width = (t_num[-1] - t_num[0]) * 0.005 if len(t_num) > 1 else 0.02
    axb.bar(t, volume, width=bar_width, color="gray", alpha=0.8)
    if ma_period and ma_period > 1:
        vol_ma = volume.rolling(window=ma_period, min_periods=1).mean()
        axb.plot(t, vol_ma, color="blue", linewidth=1.2, label="vol MA")
        axb.legend(loc="upper right")
    axb.set_ylabel("volume")
    for label in axb.get_xticklabels():
        label.set_rotation(30); label.set_ha("right")
    fig.subplots_adjust(hspace=0.15, top=0.95, bottom=0.08, left=0.08, right=0.92)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV path")
    ap.add_argument("--window_strength", type=int, default=5)
    ap.add_argument("--window_zone", type=int, default=60)
    ap.add_argument("--bins_pct", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--half_life", type=float, default=10_000_000.0)
    ap.add_argument("--quantile", type=float, default=0.8)
    ap.add_argument("--ma_period", type=int, default=10)
    ap.add_argument("--smooth", type=int, default=3)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"[X] File not found: {args.csv}")

    df = pd.read_csv(args.csv)
    dt_c = find_col(df, ["datetime","date","timestamp"])
    cl_c = find_col(df, ["close"])
    vo_c = find_col(df, ["volume","volumeusd","volumeusdt","volume(usdt)","volume usd","volume usdt"])
    if not all([dt_c, cl_c, vo_c]):
        sys.exit(f"[X] Missing required columns. Found: {list(df.columns)}")

    df = df[[dt_c, cl_c, vo_c]].copy()
    df.columns = ["datetime", "close", "volume"]
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["close","volume"])
    if df.empty:
        sys.exit("[X] No valid rows after cleaning.")

    eff_vol = compute_effective_volume(df["volume"].values, beta=args.beta, half_life=args.half_life)
    bins_df = compute_bins(df["close"].values, eff_vol, bin_pct=args.bins_pct)
    long_curve, short_curve = compute_long_short(df["close"], eff_vol, window_strength=args.window_strength,
                                                 smooth=args.smooth, window_zone=args.window_zone)

    out_root = os.path.splitext(args.csv)[0]
    out_csv = out_root + "_chip_strength.csv"
    bins_df.to_csv(out_csv, index=False)
    print(f"[OK] Saved chip_strength table -> {out_csv}")

    out_png = out_root + "_chip_timeline_pro.png"
    draw_timeline(df, bins_df, out_png, ma_period=args.ma_period, quantile=args.quantile,
                  long_curve=long_curve, short_curve=short_curve)
    print(f"[OK] Saved timeline figure -> {out_png}")

if __name__ == "__main__":
    main()
