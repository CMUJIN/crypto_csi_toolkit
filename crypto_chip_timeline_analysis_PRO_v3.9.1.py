#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Chip Timeline Analysis (v4.0 hybrid)
-------------------------------------------
Aligned with futures logic:
- Fixed-width price bins: bin_width = mean(close) * (bins_pct / 100)
- Weighted histogram per bin:
    weight = volume * (1 + max(delta_open_interest, 0))**beta * time_decay
- Two windows: ALL vs RECENT(window_zone)
- Normalize each histogram by its own max
- Keep zones with strength >= respective quantiles:
    persistent if (recent>=thr_recent) and (all>=thr_all), else short-term
- Draw highlights consistent with the futures chart style
- Keep long/short curves (from previous crypto version)

Notes:
- If 'open_interest' column is missing, delta_oi is treated as 0.
- Image/CSV text uses English to avoid encoding issues.
"""

import argparse, sys, re
import pandas as pd, numpy as np
import matplotlib.pyplot as plt


# ==============================
# Utilities
# ==============================
def fmt_price(v: float) -> str:
    """Format price: integer if >=100, else 2 decimals."""
    try:
        v = float(v)
        return f"{int(round(v))}" if v >= 100 else f"{v:.2f}"
    except Exception:
        return str(v)


def to_utc_datetime(s):
    """Safe UTC datetime parse."""
    return pd.to_datetime(s, errors="coerce", utc=True)


def exp_half_life_decay(dt_series_utc: pd.Series, half_life_days: float) -> np.ndarray:
    """
    Exponential time decay by half-life (days).
    decay(t) = exp(-ln(2) * age_days / half_life_days)
    age_days = (t_max - t) in days
    """
    t = to_utc_datetime(dt_series_utc)
    t_max = t.max()
    # if parsing fails, fallback to index-based days (hourly assumed)
    if pd.isna(t_max):
        idx = np.arange(len(dt_series_utc))[::-1]
        age_days = idx / 24.0
    else:
        age_days = (t_max - t).dt.total_seconds().values / 86400.0

    half_life_days = max(float(half_life_days), 1e-9)
    lam = np.log(2.0) / half_life_days
    return np.exp(-lam * age_days)


# ==============================
# Core: chip zones (futures logic)
# ==============================
def compute_chipzones_futures_logic(
    df: pd.DataFrame,
    window_zone: int = 60,
    bins_pct: float = 0.5,
    beta: float = 0.7,
    half_life: float = 10.0,  # in days
    quantile: float = 0.8,
):
    """
    Returns a DataFrame with:
      low, high, recent_strength, all_strength, avg_strength, zone_type, strength
    where strength = avg_strength for compatibility with Notion summary.
    """

    # Required columns
    for col in ["close", "volume", "datetime"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    close = df["close"].astype(float).values
    volume = df["volume"].astype(float).values
    dt = df["datetime"]

    # --- delta open interest (if exists) ---
    if "open_interest" in df.columns:
        oi = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0.0)
        delta_oi = oi.diff().clip(lower=0.0).fillna(0.0).values
    else:
        delta_oi = np.zeros(len(df), dtype=float)

    # --- time decay ---
    decay = exp_half_life_decay(df["datetime"], half_life_days=float(half_life))

    # --- effective weight (futures logic) ---
    # weight = volume * (1 + delta_oi)**beta * decay
    weights = volume * np.power(1.0 + delta_oi, float(beta)) * decay

    # --- fixed-width bins by mean price ---
    mean_price = np.nanmean(close)
    bw = max(mean_price * (float(bins_pct) / 100.0), 1e-12)
    pmin, pmax = float(np.nanmin(close)), float(np.nanmax(close))
    # Ensure at least 1 bin
    n_steps = max(int(np.ceil((pmax - pmin) / bw)), 1)
    bins = np.linspace(pmin, pmin + n_steps * bw, n_steps + 1)
    lows, highs = bins[:-1], bins[1:]

    def hist_norm(prices, w, _bins):
        h, _ = np.histogram(prices, bins=_bins, weights=w)
        mx = h.max() if h.size else 0.0
        return (h / mx) if mx > 0 else h

    # --- ALL window ---
    h_all = hist_norm(close, weights, bins)

    # --- RECENT window ---
    win = max(int(window_zone), 1)
    close_recent = close[-win:]
    weights_recent = weights[-win:]
    h_recent = hist_norm(close_recent, weights_recent, bins)

    # --- thresholds by quantile ---
    # Use non-zero values to compute quantile; if all zeros, set high threshold
    nonzero_all = h_all[h_all > 0]
    nonzero_rec = h_recent[h_recent > 0]
    thr_all = np.quantile(nonzero_all, quantile) if nonzero_all.size else 1.1
    thr_recent = np.quantile(nonzero_rec, quantile) if nonzero_rec.size else 1.1

    rows = []
    for i in range(len(lows)):
        ra = float(h_recent[i])
        aa = float(h_all[i])
        if (ra >= thr_recent) or (aa >= thr_all):
            avg = (ra + aa) / 2.0
            persistent = (ra >= thr_recent) and (aa >= thr_all)
            zone_type = "persistent" if persistent else "short-term"
            rows.append(
                dict(
                    low=lows[i],
                    high=highs[i],
                    recent_strength=ra,
                    all_strength=aa,
                    avg_strength=avg,
                    zone_type=zone_type,
                    strength=avg,  # for downstream compatibility
                )
            )

    if not rows:
        # if nothing passes filters, still return the strongest few
        avg_all = (h_recent + h_all) / 2.0
        order = np.argsort(-avg_all)
        k = min(10, len(order))
        for j in order[:k]:
            rows.append(
                dict(
                    low=lows[j],
                    high=highs[j],
                    recent_strength=float(h_recent[j]),
                    all_strength=float(h_all[j]),
                    avg_strength=float(avg_all[j]),
                    zone_type="short-term",
                    strength=float(avg_all[j]),
                )
            )

    chip_df = pd.DataFrame(rows)
    chip_df = chip_df.sort_values("avg_strength", ascending=False).reset_index(drop=True)
    return chip_df


# ==============================
# Long/Short curves (kept)
# ==============================
def compute_long_short(close_series: pd.Series, eff_weight: np.ndarray,
                       window_strength: int, smooth: int, window_zone: int):
    close = close_series.astype(float)
    ret = close.pct_change().fillna(0)

    vol_scale = ret.rolling(window_zone, min_periods=1).std().replace(0.0, np.nan)
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


# ==============================
# Visualization (futures-style highlight)
# ==============================
def draw_timeline(df, chip_df, out_png, ma_period, long_curve, short_curve):
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    t = to_utc_datetime(df["datetime"])
    price = df["close"].astype(float)
    volume = df["volume"].astype(float)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.4], hspace=0.15)
    ax = fig.add_subplot(gs[0, 0]); ax2 = ax.twinx()

    # --- price ---
    ax.plot(t, price, color="black", linewidth=1.2, label="price")
    ax.set_ylabel("price")
    ax2.set_ylabel("strength")

    # --- chip zones (futures-style coloring) ---
    # persistent: deeper orange; short-term: lighter
    for _, r in chip_df.iterrows():
        color = (1.0, 0.55, 0.0, 0.30) if r["zone_type"] == "persistent" else (1.0, 0.8, 0.2, 0.20)
        ax.axhspan(r["low"], r["high"], color=color)
        y_mid = (r["low"] + r["high"]) / 2.0
        label = f"{fmt_price(r['low'])}-{fmt_price(r['high'])}  strength:{r['avg_strength']:.2f}"
        ax.text(t.iloc[0], y_mid, label, fontsize=8, color="saddlebrown",
                bbox=dict(facecolor=(1, 0.95, 0.85, 0.8), edgecolor="none", boxstyle="round,pad=0.2"))

    # --- long/short curves ---
    ax2.plot(t, long_curve, color="red", linewidth=1.0, label="long")
    ax2.plot(t, short_curve, color="green", linewidth=1.0, label="short")
    ax2.set_ylim(0, 1.0)

    # legend
    ln1, = ax.plot([], [], color="black", label="price")
    ln2, = ax2.plot([], [], color="red", label="long")
    ln3, = ax2.plot([], [], color="green", label="short")
    ax.legend(handles=[ln1, ln2, ln3], loc="upper right", frameon=True)

    # --- volume subplot ---
    axb = fig.add_subplot(gs[1, 0], sharex=ax)
    axb.bar(t, volume, color="gray", alpha=0.7)
    vol_ma = volume.rolling(window=ma_period, min_periods=1).mean()
    axb.plot(t, vol_ma, color="blue", linewidth=1.2, label="vol MA")
    axb.legend(loc="upper right")
    axb.set_ylabel("volume")

    # --- y-range from min price ---
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
    ap.add_argument("csv", help="CSV file (with columns: datetime/date/timestamp, close, volume, [open_interest])")
    ap.add_argument("--window_strength", type=int, default=5)
    ap.add_argument("--window_zone", type=int, default=60)
    ap.add_argument("--bins_pct", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--half_life", type=float, default=10)  # days (futures default small, you can set large)
    ap.add_argument("--quantile", type=float, default=0.8)
    ap.add_argument("--ma_period", type=int, default=10)
    ap.add_argument("--smooth", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.lower().strip() for c in df.columns]

    # normalize columns
    rename_map = {}
    col_map = {
        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "close": ["close", "last"],
        "volume": ["volume", "volumeusd", "volume usdt"],
        "datetime": ["datetime", "date", "timestamp"]
    }
    for target, cands in col_map.items():
        for c in cands:
            if c in df.columns:
                rename_map[c] = target
                break
    df = df.rename(columns=rename_map)

    # parse datetime
    if "datetime" not in df.columns:
        raise SystemExit("[X] Missing column: datetime/date/timestamp")
    df["datetime"] = to_utc_datetime(df["datetime"])

    # filter by start date inferred from filename: *_YYYY-MM-DD_to_*
    m = re.search(r'_(\d{4}-\d{2}-\d{2})_to_', args.csv)
    if m:
        start_dt = pd.to_datetime(m.group(1), utc=True)
        df = df[df["datetime"] >= start_dt]
        print(f"[*] Filtered data from {start_dt.strftime('%Y-%m-%d')} onward, rows={len(df)}")

    for col in ["close", "volume"]:
        if col not in df.columns:
            raise SystemExit(f"[X] Missing column: {col}")

    # compute chip zones with futures logic
    chip_df = compute_chipzones_futures_logic(
        df=df,
        window_zone=args.window_zone,
        bins_pct=args.bins_pct,
        beta=args.beta,
        half_life=args.half_life,
        quantile=args.quantile,
    )

    # effective weight for long/short (reuse futures-style decay + beta on delta_oi)
    if "open_interest" in df.columns:
        oi = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0.0)
        delta_oi = oi.diff().clip(lower=0.0).fillna(0.0).values
    else:
        delta_oi = np.zeros(len(df), dtype=float)
    decay = exp_half_life_decay(df["datetime"], half_life_days=float(args.half_life))
    eff_weight = df["volume"].astype(float).values * np.power(1.0 + delta_oi, float(args.beta)) * decay

    long_curve, short_curve = compute_long_short(
        df["close"], eff_weight, args.window_strength, args.smooth, args.window_zone
    )

    # save CSV (add strength alias for compatibility)
    out_csv = args.csv.replace(".csv", "_chip_strength.csv")
    chip_out = chip_df.copy()
    if "strength" not in chip_out.columns:
        chip_out["strength"] = chip_out["avg_strength"]
    # Format price columns
    chip_out["low"] = chip_out["low"].apply(fmt_price)
    chip_out["high"] = chip_out["high"].apply(fmt_price)
    chip_out.to_csv(out_csv, index=False)
    print(f"[OK] Saved chip_strength table -> {out_csv}")

    # draw figure
    out_png = args.csv.replace(".csv", "_chip_timeline_pro.png")
    draw_timeline(df, chip_df, out_png, args.ma_period, long_curve, short_curve)


if __name__ == "__main__":
    main()
