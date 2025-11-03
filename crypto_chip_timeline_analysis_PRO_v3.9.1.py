#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Chip Timeline Analysis + Liquidity Curve (hybrid, fixed start-times)
---------------------------------------------------------------------------
Fixes:
- Liquidity uses its own start time (liquidity_start), independent of chip start
- Keep df_all (raw) for liquidity; df_chip (filtered) for chip
- Add warm-up buffer for rolling Z window before liquidity_start
"""

import argparse, re
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
    lam = np.log(2.0) / max(half_life_days, 1e-9)
    return np.exp(-lam * age_days)

def extract_symbol_from_filename(path: str) -> str:
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
# Core: chip zones
# ==============================
def compute_chipzones_futures_logic(df, window_zone=60, bins_pct=0.5, beta=0.7, half_life=10.0, quantile=0.8):
    for col in ["close", "volume", "datetime"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    close = df["close"].astype(float).values
    volume = df["volume"].astype(float).values
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
        return (h / h.max()) if h.size and h.max() > 0 else h
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
            zone_type = "persistent" if (ra >= thr_recent and aa >= thr_all) else "short-term"
            rows.append(dict(low=lows[i], high=highs[i], avg_strength=avg, zone_type=zone_type, strength=avg))
    if not rows:
        avg_all = (h_recent + h_all) / 2.0
        for j in np.argsort(-avg_all)[:10]:
            rows.append(dict(low=lows[j], high=highs[j], avg_strength=float(avg_all[j]), zone_type="short-term", strength=float(avg_all[j])))
    return pd.DataFrame(rows).sort_values("avg_strength", ascending=False).reset_index(drop=True)

# ==============================
# Long/Short Curves
# ==============================
def compute_long_short(close_series, eff_weight, window_strength, smooth, window_zone):
    close = close_series.astype(float)
    ret = close.pct_change().fillna(0)
    vol_scale = ret.rolling(window_zone, min_periods=1).std().replace(0.0, np.nan).fillna(ret.std() or 1.0)
    shifted_close = close.shift(1).bfill()
    impulse = (close.diff() / (vol_scale * shifted_close)) * (eff_weight / (np.nanmax(eff_weight) or 1.0))
    impulse = impulse.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    impulse_s = impulse.rolling(window_strength, min_periods=1).mean()
    long_raw, short_raw = np.clip(impulse_s, 0, None), np.clip(-impulse_s, 0, None)
    norm = lambda x: (x / (x.max() or 1.0)).rolling(smooth, min_periods=1).mean().fillna(0.0)
    return norm(long_raw), norm(short_raw)

# ==============================
# Visualization (Chip)
# ==============================
def draw_timeline(df, chip_df, out_png, ma_period, long_curve, short_curve):
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    t = to_utc_datetime(df["datetime"])
    price, volume = df["close"].astype(float), df["volume"].astype(float)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.4], hspace=0.15)
    ax = fig.add_subplot(gs[0, 0]); ax2 = ax.twinx()
    ax.plot(t, price, color="black", linewidth=1.2, label="price")
    ax2.plot(t, long_curve, color="red", linewidth=1.0, label="long")
    ax2.plot(t, short_curve, color="green", linewidth=1.0, label="short")
    ax2.set_ylim(0, 1.0)
    for _, r in chip_df.iterrows():
        color = (1, 0.55, 0, 0.3) if r["zone_type"] == "persistent" else (1, 0.8, 0.2, 0.2)
        ax.axhspan(r["low"], r["high"], color=color)
    ax.legend(loc="upper right"); ax2.legend(loc="lower right")
    axb = fig.add_subplot(gs[1, 0], sharex=ax)
    axb.bar(t, volume, color="gray", alpha=0.7)
    axb.plot(t, volume.rolling(ma_period, min_periods=1).mean(), color="blue", linewidth=1.2)
    plt.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] Saved chip timeline -> {out_png}")

# ==============================
# Liquidity Section (fixed start)
# ==============================
def detect_quote_volume_column(df):
    for c in df.columns:
        if ("quote" in c.lower() and "vol" in c.lower()) or ("usdt" in c.lower() and "volume" in c.lower()):
            return c
    return None

def compute_liquidity_series(df, span_hours=24, zwin_days=90):
    price, vol = df["close"].astype(float), df["volume"].astype(float)
    qv_col = detect_quote_volume_column(df)
    dollar_vol = pd.to_numeric(df[qv_col], errors="coerce") if qv_col else price * vol
    if "high" in df.columns and "low" in df.columns:
        high, low = pd.to_numeric(df["high"], errors="coerce"), pd.to_numeric(df["low"], errors="coerce")
        prev_close = price.shift(1)
        tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    else:
        tr = (price.pct_change().abs() * price).fillna(0)
    dv_ewm, tr_ewm = ewma(dollar_vol, span_hours), ewma(tr.replace(0, np.nan).fillna(tr.median()), span_hours)
    lp_hr = (dv_ewm / tr_ewm).replace([np.inf, -np.inf], np.nan)
    return zscore_rolling(lp_hr, int(24*zwin_days)), price

def draw_liquidity_curves(df_hourly, symbol, out_csv, out_png, span_hours=24, zwin_days=90, start_dt=None):
    # 保证有 DatetimeIndex
    df = df_hourly.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    # 计算期：为保证滚动Z分正确，引入预热窗口
    if start_dt is not None:
        warmup_start = (pd.to_datetime(start_dt, utc=True) - pd.Timedelta(days=zwin_days + 7))
        df_compute = df[df.index >= warmup_start]
    else:
        df_compute = df

    liq_z, price = compute_liquidity_series(df_compute.reset_index(), span_hours, zwin_days)
    liq_z.index = df_compute.index
    price.index = df_compute.index

    # 输出期：严格从 liquidity_start 开始
    if start_dt is not None:
        liq_z = liq_z[liq_z.index >= pd.to_datetime(start_dt, utc=True)]
        price = price[price.index >= pd.to_datetime(start_dt, utc=True)]

    # 重采样
    daily = pd.DataFrame({
        "Liquidity": liq_z.resample("1D").mean(),
        "Price": price.resample("1D").last()
    }).dropna()
    weekly = pd.DataFrame({
        "Liquidity": liq_z.resample("1W").mean(),
        "Price": price.resample("1W").last()
    }).dropna()

    # 保存
    pd.DataFrame({"LP_HR_Z": liq_z, "Price": price}).dropna().to_csv(out_csv)
    print(f"[OK] Saved liquidity table -> {out_csv}")

    # 绘图
    fig = plt.figure(figsize=(12,8))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(weekly.index, weekly["Price"], color="tab:red", label="Price", alpha=0.8)
    ax1b = ax1.twinx()
    ax1b.plot(weekly.index, weekly["Liquidity"], color="tab:blue", label="Liquidity (Z)")
    for y in [-2,-1,0,1,2]: ax1b.axhline(y=y, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax1.set_title(f"{symbol} Weekly Price vs Liquidity Z")
    ax1.legend(loc="upper left"); ax1b.legend(loc="upper right")

    ax2 = plt.subplot(2,1,2)
    tail = daily.iloc[-90:]
    ax2.plot(tail.index, tail["Price"], color="tab:red", label="Price", alpha=0.8)
    ax2b = ax2.twinx()
    ax2b.plot(tail.index, tail["Liquidity"], color="tab:blue", label="Liquidity (Z)")
    for y in [-2,-1,0,1,2]: ax2b.axhline(y=y, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax2.set_title(f"{symbol} Daily Liquidity (90d)")
    ax2.legend(loc="upper left"); ax2b.legend(loc="upper right")
    plt.tight_layout(); fig.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] Saved liquidity figure -> {out_png}")

# ==============================
# Main
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV path")
    ap.add_argument("--window_strength", type=int, default=5)
    ap.add_argument("--window_zone", type=int, default=60)
    ap.add_argument("--bins_pct", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--half_life", type=float, default=10)
    ap.add_argument("--quantile", type=float, default=0.8)
    ap.add_argument("--ma_period", type=int, default=10)
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--liquidity_start", type=str, help="YYYY-MM-DD")
    ap.add_argument("--out_dir", type=str, default="docs")
    ap.add_argument("--liquidity_span_hours", type=int, default=24)
    ap.add_argument("--liquidity_zwindow_days", type=int, default=90)
    args = ap.parse_args()

    # 读取并标准化
    df_all = pd.read_csv(args.csv)
    df_all.columns = [c.lower().strip() for c in df_all.columns]
    rename = {}
    col_map = {"open": ["open"], "high": ["high"], "low": ["low"], "close": ["close","last"],
               "volume":["volume","volumeusd","volume usdt"], "datetime":["datetime","date","timestamp"]}
    for k,v in col_map.items():
        for c in v:
            if c in df_all.columns: rename[c]=k; break
    df_all = df_all.rename(columns=rename)
    if "datetime" not in df_all.columns:
        raise SystemExit("[X] Missing datetime column")
    df_all["datetime"] = to_utc_datetime(df_all["datetime"])

    # df_chip：仅用于筹码分析，保留原有“按文件名起始时间”过滤逻辑
    df_chip = df_all.copy()
    m = re.search(r'_(\d{4}-\d{2}-\d{2})_to_', args.csv)
    if m:
        chip_start = pd.to_datetime(m.group(1), utc=True)
        df_chip = df_chip[df_chip["datetime"] >= chip_start]
        print(f"[*] Chip filter from {chip_start.strftime('%Y-%m-%d')}, rows={len(df_chip)}")

    # ---- Chip analysis ----
    chip_df = compute_chipzones_futures_logic(df_chip, args.window_zone, args.bins_pct, args.beta, args.half_life, args.quantile)
    if "open_interest" in df_chip.columns:
        oi = pd.to_numeric(df_chip["open_interest"], errors="coerce").fillna(0)
        delta_oi = oi.diff().clip(lower=0).fillna(0).values
    else:
        delta_oi = np.zeros(len(df_chip))
    decay = exp_half_life_decay(df_chip["datetime"], args.half_life)
    eff_weight = df_chip["volume"].astype(float).values * np.power(1 + delta_oi, args.beta) * decay
    long_curve, short_curve = compute_long_short(df_chip["close"], eff_weight, args.window_strength, args.smooth, args.window_zone)

    chip_csv = args.csv.replace(".csv","_chip_strength.csv")
    chip_df.to_csv(chip_csv, index=False)
    chip_png = args.csv.replace(".csv","_chip_timeline_pro.png")
    draw_timeline(df_chip, chip_df, chip_png, args.ma_period, long_curve, short_curve)

    # ---- Liquidity analysis ----
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    symbol = extract_symbol_from_filename(args.csv)
    liq_csv = out_dir / f"{symbol}_liquidity_curves.csv"
    liq_png = out_dir / f"{symbol}_liquidity_curves.png"
    liq_start = pd.to_datetime(args.liquidity_start, utc=True) if args.liquidity_start else None

    # 注意：传入未截断的 df_all，由 draw_liquidity_curves 内部依据 liquidity_start + 预热窗口自行裁剪
    cols = ["datetime","close","volume"]
    if all(c in df_all.columns for c in ["open","high","low"]):
        cols = ["datetime","open","high","low","close","volume"]
    draw_liquidity_curves(df_all[cols].copy(), symbol, liq_csv, liq_png,
                          args.liquidity_span_hours, args.liquidity_zwindow_days, liq_start)

if __name__=="__main__":
    main()
