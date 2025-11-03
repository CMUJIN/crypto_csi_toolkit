#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crypto_chip_timeline_analysis_PRO_v3.9.1_fixed.py
-------------------------------------------------
Hybrid version with Liquidity Curve integration (DatetimeIndex fixed)

Functions:
- Chip zone analysis (futures logic)
- Long/Short momentum curves
- Liquidity curve (price vs liquidity z-score)
- Compatible with Notion uploader
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

def ewma(series, span):
    return series.ewm(span=span, adjust=False, min_periods=max(span//2,1)).mean()

def zscore_rolling(series, window):
    m = series.rolling(window, min_periods=max(window//10,1)).mean()
    s = series.rolling(window, min_periods=max(window//10,1)).std()
    return (series - m) / s

def extract_symbol_from_filename(path: str) -> str:
    stem = Path(path).stem.upper()
    m = re.search(r'([A-Z]{3,5})USDT', stem)
    if m:
        return m.group(1)
    parts = stem.split("_")
    return parts[1][:5] if len(parts) >= 2 else stem[:5]


# ==============================
# Core: chip zones (existing)
# ==============================
def compute_chipzones_futures_logic(df: pd.DataFrame,
                                    window_zone: int = 60,
                                    bins_pct: float = 0.5,
                                    beta: float = 0.7,
                                    half_life: float = 10.0,
                                    quantile: float = 0.8):
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
        ra = float(h_recent[i])
        aa = float(h_all[i])
        if (ra >= thr_recent) or (aa >= thr_all):
            avg = (ra + aa) / 2.0
            zone_type = "persistent" if ((ra >= thr_recent) and (aa >= thr_all)) else "short-term"
            rows.append(dict(
                low=lows[i], high=highs[i],
                recent_strength=ra, all_strength=aa,
                avg_strength=avg, zone_type=zone_type, strength=avg
            ))

    if not rows:
        avg_all = (h_recent + h_all) / 2.0
        order = np.argsort(-avg_all)
        for j in order[:min(10, len(order))]:
            rows.append(dict(
                low=lows[j], high=highs[j],
                recent_strength=float(h_recent[j]),
                all_strength=float(h_all[j]),
                avg_strength=float(avg_all[j]),
                zone_type="short-term", strength=float(avg_all[j])
            ))

    chip_df = pd.DataFrame(rows).sort_values("avg_strength", ascending=False).reset_index(drop=True)
    return chip_df


# ==============================
# Long/Short curves
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
# Liquidity computation
# ==============================
def detect_quote_volume_column(df: pd.DataFrame):
    for c in df.columns:
        cl = c.lower()
        if ("quote" in cl and "vol" in cl) or ("usdt" in cl and "volume" in cl):
            return c
    return None

def compute_liquidity_series(df: pd.DataFrame, span_hours: int = 24, zwin_days: int = 90):
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("Liquidity needs columns: close, volume")
    price = df["close"].astype(float)
    vol = df["volume"].astype(float)
    qv_col = detect_quote_volume_column(df)
    dollar_vol = pd.to_numeric(df[qv_col], errors="coerce") if qv_col else price * vol
    high = pd.to_numeric(df["high"], errors="coerce") if "high" in df.columns else None
    low = pd.to_numeric(df["low"], errors="coerce") if "low" in df.columns else None
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


def draw_liquidity_curves(df_hourly: pd.DataFrame, symbol: str,
                          out_csv: Path, out_png: Path,
                          span_hours: int = 24, zwin_days: int = 90,
                          start_dt: pd.Timestamp | None = None):
    df_hourly = df_hourly.copy()
    df_hourly["datetime"] = to_utc_datetime(df_hourly["datetime"])
    df_hourly = df_hourly.set_index("datetime").sort_index()
    if start_dt is not None:
        df_hourly = df_hourly[df_hourly.index >= start_dt]

    liq_z, price = compute_liquidity_series(df_hourly, span_hours, zwin_days)

    daily = pd.DataFrame({"Liquidity": liq_z.resample("1D").mean(),
                          "Price": price.resample("1D").last()}).dropna()
    weekly = pd.DataFrame({"Liquidity": liq_z.resample("1W").mean(),
                           "Price": price.resample("1W").last()}).dropna()

    enriched = pd.DataFrame({"LP_HR_Z": liq_z, "Price": price}).dropna()
    enriched.to_csv(out_csv)
    print(f"[OK] Saved liquidity table -> {out_csv}")

    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(weekly.index, weekly["Price"], color="tab:red", label="Price", alpha=0.7)
    ax1.set_ylabel("Price", color="tab:red")
    ax1b = ax1.twinx()
    ax1b.plot(weekly.index, weekly["Liquidity"], color="tab:blue", label="Liquidity Z")
    for y in [-2, -1, 0, 1, 2]:
        ax1b.axhline(y, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_title(f"{symbol} Weekly Price vs Liquidity Z-Score")
    ax1.legend(loc="upper left"); ax1b.legend(loc="upper right")

    ax2 = plt.subplot(2, 1, 2)
    tail = daily.iloc[-90:]
    ax2.plot(tail.index, tail["Price"], color="tab:red", label="Price", alpha=0.7)
    ax2b = ax2.twinx()
    ax2b.plot(tail.index, tail["Liquidity"], color="tab:blue", label="Liquidity Z")
    for y in [-2, -1, 0, 1, 2]:
        ax2b.axhline(y, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_title(f"{symbol} Daily Price vs Liquidity Z (Last 90 Days)")
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
    ap.add_argument("csv")
    ap.add_argument("--window_strength", type=int, default=5)
    ap.add_argument("--window_zone", type=int, default=60)
    ap.add_argument("--bins_pct", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--half_life", type=float, default=10)
    ap.add_argument("--quantile", type=float, default=0.8)
    ap.add_argument("--ma_period", type=int, default=10)
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--liquidity_start", type=str)
    ap.add_argument("--out_dir", type=str, default="docs")
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
            if c in df.columns:
                rename_map[c] = target; break
    df = df.rename(columns=rename_map)
    df["datetime"] = to_utc_datetime(df["datetime"])

    # chip zone
    chip_df = compute_chipzones_futures_logic(df, args.window_zone,
                                              args.bins_pct, args.beta,
                                              args.half_life, args.quantile)
    if "open_interest" in df.columns:
        oi = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0.0)
        delta_oi = oi.diff().clip(lower=0.0).fillna(0.0).values
    else:
        delta_oi = np.zeros(len(df), dtype=float)
    decay = exp_half_life_decay(df["datetime"], args.half_life)
    eff_weight = df["volume"].astype(float).values * np.power(1.0 + delta_oi, args.beta) * decay
    long_curve, short_curve = compute_long_short(df["close"], eff_weight,
                                                 args.window_strength, args.smooth, args.window_zone)
    chip_csv = args.csv.replace(".csv", "_chip_strength.csv")
    chip_df.to_csv(chip_csv, index=False)
    print(f"[OK] Saved chip_strength table -> {chip_csv}")

    chip_png = args.csv.replace(".csv", "_chip_timeline_pro.png")
    symbol = extract_symbol_from_filename(args.csv)
    print(f"[*] Processing liquidity for {symbol}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    liq_start = pd.to_datetime(args.liquidity_start, utc=True) if args.liquidity_start else None
    liq_start_str = args.liquidity_start if args.liquidity_start else "FULL"
    liq_csv = out_dir / f"{symbol}_liquidity_{liq_start_str}.csv"
    liq_png = out_dir / f"{symbol}_liquidity_{liq_start_str}.png"
    draw_liquidity_curves(df, symbol, liq_csv, liq_png,
                          span_hours=24, zwin_days=90, start_dt=liq_start)

if __name__ == "__main__":
    main()
