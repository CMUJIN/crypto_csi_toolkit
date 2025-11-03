#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Chip Timeline Analysis + Liquidity Curve (hybrid)
-------------------------------------------------------
This file extends your original chip analysis with a liquidity curve:
- Left axis: Price (red)
- Right axis: Liquidity Score Z (blue) with horizontal dashed lines (-2,-1,0,1,2)
- Title includes symbol extracted from CSV name (e.g., BTC)

Outputs (by default under docs/):
- docs/{SYMBOL}_liquidity_curves.csv
- docs/{SYMBOL}_liquidity_curves.png

Chip (existing) outputs kept as-is:
- *_chip_strength.csv
- *_chip_timeline_pro.png
"""

import argparse, sys, re, os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# Utilities
# ==============================
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

def extract_symbol_from_filename(path: str) -> str:
    """
    Try to get SYMBOL from name like Binance_BTCUSDT_1h_*.csv -> BTC;
    fallback to the second token split by '_' or stem.
    """
    stem = Path(path).stem.upper()
    m = re.search(r'([A-Z]{3,5})USDT', stem)
    if m: return m.group(1)
    parts = stem.split("_")
    return parts[1][:5] if len(parts) >= 2 else stem[:5]

def ewma(series, span):
    return series.ewm(span=span, adjust=False, min_periods=max(span//2,1)).mean()

def zscore_rolling(series, window):
    m = series.rolling(window, min_periods=max(window//10,1)).mean()
    s = series.rolling(window, min_periods=max(window//10,1)).std()
    return (series - m) / s

# ==============================
# Core: chip zones (existing)
# ==============================
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

    if "open_interest" in df.columns:
        oi = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0.0)
        delta_oi = oi.diff().clip(lower=0.0).fillna(0.0).values
    else:
        delta_oi = np.zeros(len(df), dtype=float)

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

    nonzero_all = h_all[h_all > 0]
    nonzero_rec = h_recent[h_recent > 0]
    thr_all = np.quantile(nonzero_all, quantile) if nonzero_all.size else 1.1
    thr_recent = np.quantile(nonzero_rec, quantile) if nonzero_rec.size else 1.1

    rows = []
    for i in range(len(lows)):
        ra = float(h_recent[i]); aa = float(h_all[i])
        if (ra >= thr_recent) or (aa >= thr_all):
            avg = (ra + aa) / 2.0
            zone_type = "persistent" if ((ra >= thr_recent) and (aa >= thr_all)) else "short-term"
            rows.append(dict(
                low=lows[i], high=highs[i], recent_strength=ra, all_strength=aa,
                avg_strength=avg, zone_type=zone_type, strength=avg
            ))

    if not rows:
        avg_all = (h_recent + h_all) / 2.0
        order = np.argsort(-avg_all)
        for j in order[:min(10, len(order))]:
            rows.append(dict(
                low=lows[j], high=highs[j], recent_strength=float(h_recent[j]),
                all_strength=float(h_all[j]), avg_strength=float(avg_all[j]),
                zone_type="short-term", strength=float(avg_all[j])
            ))

    chip_df = pd.DataFrame(rows).sort_values("avg_strength", ascending=False).reset_index(drop=True)
    return chip_df

# ==============================
# Long/Short curves (existing)
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
# Visualization (existing chip)
# ==============================
def draw_timeline(df, chip_df, out_png, ma_period, long_curve, short_curve):
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    t = to_utc_datetime(df["datetime"])
    price = df["close"].astype(float)
    volume = df["volume"].astype(float)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.4], hspace=0.15)
    ax = fig.add_subplot(gs[0, 0]); ax2 = ax.twinx()

    ax.plot(t, price, color="black", linewidth=1.2, label="price")
    ax.set_ylabel("price")
    ax2.set_ylabel("strength")

    for _, r in chip_df.iterrows():
        color = (1.0, 0.55, 0.0, 0.30) if r["zone_type"] == "persistent" else (1.0, 0.8, 0.2, 0.20)
        ax.axhspan(r["low"], r["high"], color=color)
        y_mid = (r["low"] + r["high"]) / 2.0
        label = f"{fmt_price(r['low'])}-{fmt_price(r['high'])}  strength:{r['avg_strength']:.2f}"
        ax.text(t.iloc[0], y_mid, label, fontsize=8, color="saddlebrown",
                bbox=dict(facecolor=(1, 0.95, 0.85, 0.8), edgecolor="none", boxstyle="round,pad=0.2"))

    ax2.plot(t, long_curve, color="red", linewidth=1.0, label="long")
    ax2.plot(t, short_curve, color="green", linewidth=1.0, label="short")
    ax2.set_ylim(0, 1.0)

    ln1, = ax.plot([], [], color="black", label="price")
    ln2, = ax2.plot([], [], color="red", label="long")
    ln3, = ax2.plot([], [], color="green", label="short")
    ax.legend(handles=[ln1, ln2, ln3], loc="upper right", frameon=True)

    axb = fig.add_subplot(gs[1, 0], sharex=ax)
    axb.bar(t, volume, color="gray", alpha=0.7)
    vol_ma = volume.rolling(window=ma_period, min_periods=1).mean()
    axb.plot(t, vol_ma, color="blue", linewidth=1.2, label="vol MA")
    axb.legend(loc="upper right")
    axb.set_ylabel("volume")

    ymin, ymax = price.min(), price.max()
    ax.set_ylim(ymin * 0.98, ymax * 1.02)

    for label in axb.get_xticklabels():
        label.set_rotation(30); label.set_ha("right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved timeline figure -> {out_png}")

# ==============================
# NEW: Liquidity computation & plot
# ==============================
def detect_quote_volume_column(df: pd.DataFrame):
    for c in df.columns:
        cl = c.lower()
        if ("quote" in cl and "vol" in cl) or ("usdt" in cl and "volume" in cl):
            return c
    return None

def compute_liquidity_series(df: pd.DataFrame,
                             span_hours: int = 24,
                             zwin_days: int = 90):
    """
    Return hourly LiquidityScore_Z and price series (indexed by datetime).
    Liquidity = Zscore( EWMA( DollarVolume / TR ) ).
    If quote volume exists, use it; else use close*volume proxy.
    """
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("Liquidity needs columns: close, volume")

    price = df["close"].astype(float)
    vol = df["volume"].astype(float)

    qv_col = detect_quote_volume_column(df)
    if qv_col is not None:
        dollar_vol = pd.to_numeric(df[qv_col], errors="coerce")
    else:
        dollar_vol = price * vol

    # True Range (robust fallback)
    high = pd.to_numeric(df["high"], errors="coerce") if "high" in df.columns else None
    low  = pd.to_numeric(df["low"], errors="coerce") if "low" in df.columns else None
    if high is not None and low is not None:
        prev_close = price.shift(1)
        tr = pd.concat([(high - low).abs(),
                        (high - prev_close).abs(),
                        (low - prev_close).abs()], axis=1).max(axis=1)
    else:
        tr = (price.pct_change().abs() * price).fillna(0)

    dv_ewm = ewma(dollar_vol, span_hours)
    tr_ewm = ewma(tr.replace(0, np.nan).fillna(tr.median()), span_hours)
    lp_hr = (dv_ewm / tr_ewm).replace([np.inf, -np.inf], np.nan)

    zwin = max(int(24 * zwin_days), 24)
    liquidity_z = zscore_rolling(lp_hr, window=zwin)

    return liquidity_z, price

def draw_liquidity_curves(df_hourly: pd.DataFrame,
                          symbol: str,
                          out_csv: Path,
                          out_png: Path,
                          span_hours: int = 24,
                          zwin_days: int = 90,
                          start_dt: pd.Timestamp | None = None):
    if start_dt is not None:
        df_hourly = df_hourly[df_hourly["datetime"] >= start_dt]

    liq_z, price = compute_liquidity_series(df_hourly, span_hours, zwin_days)

    # Resample
    daily = pd.DataFrame({"Liquidity": liq_z.resample("1D").mean(),
                          "Price": price.resample("1D").last()}).dropna()
    weekly = pd.DataFrame({"Liquidity": liq_z.resample("1W").mean(),
                           "Price": price.resample("1W").last()}).dropna()

    # Save CSV
    enriched = pd.DataFrame({
        "LP_HR_Z": liq_z,
        "Price": price
    }).dropna()
    enriched.to_csv(out_csv)
    print(f"[OK] Saved liquidity table -> {out_csv}")

    # Plot
    fig = plt.figure(figsize=(12, 8))

    # Weekly
    ax1 = plt.subplot(2,1,1)
    ax1.plot(weekly.index, weekly["Price"], color="tab:red", label="Price", alpha=0.7)
    ax1.set_ylabel("Price", color="tab:red")
    ax1.tick_params(axis='y', labelcolor="tab:red")
    ax1b = ax1.twinx()
    ax1b.plot(weekly.index, weekly["Liquidity"], color="tab:blue", label="Liquidity Score (Z)")
    ax1b.set_ylabel("Liquidity Z-Score", color="tab:blue")
    ax1b.tick_params(axis='y', labelcolor="tab:blue")
    for y in [-2, -1, 0, 1, 2]:
        ax1b.axhline(y=y, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.set_title(f"{symbol} Weekly Price vs Liquidity Score")
    ax1.legend(loc="upper left"); ax1b.legend(loc="upper right")

    # Daily (last 90 days)
    ax2 = plt.subplot(2,1,2)
    tail = daily.iloc[-90:]
    ax2.plot(tail.index, tail["Price"], color="tab:red", label="Price", alpha=0.7)
    ax2.set_ylabel("Price", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    ax2b = ax2.twinx()
    ax2b.plot(tail.index, tail["Liquidity"], color="tab:blue", label="Liquidity Score (Z)")
    ax2b.set_ylabel("Liquidity Z-Score", color="tab:blue")
    ax2b.tick_params(axis='y', labelcolor="tab:blue")
    for y in [-2, -1, 0, 1, 2]:
        ax2b.axhline(y=y, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax2.set_title(f"{symbol} Daily Price vs Liquidity Score (Last 90 Days)")
    ax2.legend(loc="upper left"); ax2b.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved liquidity figure -> {out_png}")

# ==============================
# Main
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV file (with columns: datetime/date/timestamp, close, volume, [open_interest])")
    # Existing chip args
    ap.add_argument("--window_strength", type=int, default=5)
    ap.add_argument("--window_zone", type=int, default=60)
    ap.add_argument("--bins_pct", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--half_life", type=float, default=10)
    ap.add_argument("--quantile", type=float, default=0.8)
    ap.add_argument("--ma_period", type=int, default=10)
    ap.add_argument("--smooth", type=int, default=3)
    # NEW liquidity args (to match your config.yaml needs)
    ap.add_argument("--liquidity_start", type=str, help="Start date for liquidity curve (YYYY-MM-DD)")
    ap.add_argument("--out_dir", type=str, default="docs", help="Output directory for figures/CSVs")
    ap.add_argument("--liquidity_span_hours", type=int, default=24, help="EWMA span hours for LP")
    ap.add_argument("--liquidity_zwindow_days", type=int, default=90, help="Rolling z-score window in days")
    args = ap.parse_args()

    # ----- Load & normalize columns
    df = pd.read_csv(args.csv)
    df.columns = [c.lower().strip() for c in df.columns]
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
    if "datetime" not in df.columns:
        raise SystemExit("[X] Missing column: datetime/date/timestamp")
    df["datetime"] = to_utc_datetime(df["datetime"])

    # Optional filename-based filter (kept from your code)
    m = re.search(r'_(\d{4}-\d{2}-\d{2})_to_', args.csv)
    if m:
        start_dt_from_name = pd.to_datetime(m.group(1), utc=True)
        df = df[df["datetime"] >= start_dt_from_name]
        print(f"[*] Filtered data from {start_dt_from_name.strftime('%Y-%m-%d')} onward, rows={len(df)}")

    for col in ["close", "volume"]:
        if col not in df.columns:
            raise SystemExit(f"[X] Missing column: {col}")

    # ----- Chip analysis (unchanged)
    chip_df = compute_chipzones_futures_logic(
        df=df,
        window_zone=args.window_zone,
        bins_pct=args.bins_pct,
        beta=args.beta,
        half_life=args.half_life,
        quantile=args.quantile,
    )

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

    chip_csv = args.csv.replace(".csv", "_chip_strength.csv")
    chip_out = chip_df.copy()
    if "strength" not in chip_out.columns:
        chip_out["strength"] = chip_out["avg_strength"]
    chip_out["low"] = chip_out["low"].apply(fmt_price)
    chip_out["high"] = chip_out["high"].apply(fmt_price)
    chip_out.to_csv(chip_csv, index=False)
    print(f"[OK] Saved chip_strength table -> {chip_csv}")

    chip_png = args.csv.replace(".csv", "_chip_timeline_pro.png")
    draw_timeline(df, chip_df, chip_png, args.ma_period, long_curve, short_curve)

    # ----- Liquidity curve (NEW)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbol = extract_symbol_from_filename(args.csv)

    liq_start = pd.to_datetime(args.liquidity_start, utc=True) if args.liquidity_start else None
    liq_csv = out_dir / f"{symbol}_liquidity_curves.csv"
    liq_png = out_dir / f"{symbol}_liquidity_curves.png"

    draw_liquidity_curves(
        df_hourly=df[["datetime","open","high","low","close","volume"] if "open" in df.columns and "high" in df.columns and "low" in df.columns else ["datetime","close","volume"]].copy(),
        symbol=symbol,
        out_csv=liq_csv,
        out_png=liq_png,
        span_hours=args.liquidity_span_hours,
        zwin_days=args.liquidity_zwindow_days,
        start_dt=liq_start
    )

if __name__ == "__main__":
    main()
