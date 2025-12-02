#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crypto_chip_timeline_analysis_PRO_v3.9.1_fixed.py
-------------------------------------------------
✅ 保留原始筹码分析逻辑
✅ 图片输出增加时间戳，避免 Notion / CDN 缓存问题：
    - docs/{SYMBOL}_chip_strength.csv          （CSV 不带时间戳）
    - docs/{SYMBOL}_chip_timeline_pro_YYYYMMDD_HH.png
    - docs/{SYMBOL}_liquidity_YYYY-MM-DD_YYYYMMDD_HH.png
✅ 流动性曲线（严格使用 --liquidity_start 截取数据）：
    - 左轴：Price
    - 右轴：Liquidity Z-Score（-2, -1, 0, 1, 2 参考线）
    - 上图：全样本周线；下图：最近90天日线
✅ 与 upload_to_notion.py 可通过通配符匹配最新图片（*_chip_timeline_pro*.png、*_liquidity_*.png）
"""

import argparse
import os
import re
from pathlib import Path
from datetime import datetime
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Utils
# =========================================================
def fmt_price(v: float) -> str:
    """>=100 取整，否则保留两位小数"""
    try:
        v = float(v)
        return f"{int(round(v))}" if v >= 100 else f"{v:.2f}"
    except Exception:
        return str(v)


def to_utc_datetime(s):
    """安全的 UTC 时间解析"""
    return pd.to_datetime(s, errors="coerce", utc=True)


def exp_half_life_decay(dt_series_utc: pd.Series, half_life_days: float) -> np.ndarray:
    """
    指数半衰期衰减（天）
    decay(t) = exp(-ln(2) * age_days / half_life_days)
    """
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
    从文件名中鲁棒提取 SYMBOL：
    例：Binance_AAVEUSDT_1h_2025-10-02_to_latest.csv -> AAVEUSDT
    若无 USDT/ USD，则回退第二段或首段。
    """
    stem = Path(path).stem
    parts = stem.split("_")
    for p in parts:
        up = p.upper()
        if "USDT" in up or (up.endswith("USD") and len(up) >= 6):
            return up
    # fallback
    if len(parts) >= 2:
        return parts[1].upper()
    return stem.upper()[:12]


def ensure_docs_dir():
    Path("docs").mkdir(parents=True, exist_ok=True)


# === 删除旧图片（防止目录里堆历史文件 & Notion 乱指向）===
def delete_old_chip_images(symbol: str, out_dir: str):
    """
    删除当前 symbol 相关的旧 PNG：
      - {SYMBOL}_chip_timeline_pro*.png
    """
    patterns = [
        os.path.join(out_dir, f"{symbol}_chip_timeline_pro*.png"),
    ]
    for pattern in patterns:
        for fp in glob.glob(pattern):
            try:
                os.remove(fp)
                print(f"[INFO] 删除旧筹码图：{fp}")
            except Exception as e:
                print(f"[WARN] 删除旧筹码图失败 {fp}: {e}")


def delete_old_liquidity_images(symbol: str, out_dir: str):
    """
    删除当前 symbol 相关的旧流动性 PNG：
      - {SYMBOL}_liquidity_*.png
    """
    pattern = os.path.join(out_dir, f"{symbol}_liquidity_*.png")
    for fp in glob.glob(pattern):
        try:
            os.remove(fp)
            print(f"[INFO] 删除旧流动性图：{fp}")
        except Exception as e:
            print(f"[WARN] 删除旧流动性图失败 {fp}: {e}")


# =========================================================
# Chip zones (futures logic)
# =========================================================
def compute_chipzones_futures_logic(
    df: pd.DataFrame,
    window_zone: int = 60,
    bins_pct: float = 0.5,
    beta: float = 0.7,
    half_life: float = 10.0,  # days
    quantile: float = 0.8,
):
    """
    返回列：
      low, high, recent_strength, all_strength, avg_strength, zone_type, strength
    """
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

    # ALL
    h_all = hist_norm(close, weights, bins)
    # RECENT
    win = max(int(window_zone), 1)
    h_recent = hist_norm(close[-win:], weights[-win:], bins)

    nz_all = h_all[h_all > 0]
    nz_rec = h_recent[h_recent > 0]
    thr_all = np.quantile(nz_all, quantile) if nz_all.size else 1.1
    thr_recent = np.quantile(nz_rec, quantile) if nz_rec.size else 1.1

    rows = []
    for i in range(len(lows)):
        ra = float(h_recent[i])
        aa = float(h_all[i])
        if (ra >= thr_recent) or (aa >= thr_all):
            avg = (ra + aa) / 2.0
            zone_type = "persistent" if ((ra >= thr_recent) and (aa >= thr_all)) else "short-term"
            rows.append(
                dict(
                    low=lows[i],
                    high=highs[i],
                    recent_strength=ra,
                    all_strength=aa,
                    avg_strength=avg,
                    zone_type=zone_type,
                    strength=avg,
                )
            )

    if not rows:
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

    chip_df = pd.DataFrame(rows).sort_values("avg_strength", ascending=False).reset_index(drop=True)
    return chip_df


# =========================================================
# Long / Short curves
# =========================================================
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


# =========================================================
# Draw: Chip timeline
# =========================================================
def draw_timeline(df, chip_df, out_png, ma_period, long_curve, short_curve):
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    t = to_utc_datetime(df["datetime"])
    price = df["close"].astype(float)
    volume = df["volume"].astype(float)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.4], hspace=0.15)
    ax = fig.add_subplot(gs[0, 0]); ax2 = ax.twinx()

    # price
    ax.plot(t, price, color="black", linewidth=1.2, label="price")
    ax.set_ylabel("price")
    ax2.set_ylabel("strength")

    # chip zones
    for _, r in chip_df.iterrows():
        color = (1.0, 0.55, 0.0, 0.30) if r["zone_type"] == "persistent" else (1.0, 0.8, 0.2, 0.20)
        ax.axhspan(r["low"], r["high"], color=color)

    # long / short
    ax2.plot(t, long_curve, color="red", linewidth=1.0, label="long")
    ax2.plot(t, short_curve, color="green", linewidth=1.0, label="short")
    ax2.set_ylim(0, 1.0)

    # legend
    ln1, = ax.plot([], [], color="black", label="price")
    ln2, = ax2.plot([], [], color="red", label="long")
    ln3, = ax2.plot([], [], color="green", label="short")
    ax.legend(handles=[ln1, ln2, ln3], loc="upper right", frameon=True)

    # volume subplot
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


# =========================================================
# Liquidity helpers
# =========================================================
def detect_quote_volume_column(df: pd.DataFrame):
    """优先选择法币/报价成交额列"""
    for c in df.columns:
        cl = c.lower().replace(" ", "")
        if cl in ("volumeusdt", "volumeusd", "quotevolume", "quoteassetvolume"):
            return c
    return None


def ewma(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(span // 2, 1)).mean()


def zscore_rolling(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window, min_periods=max(window // 10, 1)).mean()
    s = series.rolling(window, min_periods=max(window // 10, 1)).std()
    return (series - m) / s


def compute_liquidity_series(
    df: pd.DataFrame,
    span_hours: int = 24,
    zwin_days: int = 90,
) -> tuple[pd.Series, pd.Series]:
    """
    Liquidity = Zscore( EWMA( DollarVolume / TrueRange ) )
    返回：hourly 的 (liq_z, price)，index 均为 DatetimeIndex
    """
    if "datetime" not in df.columns or "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("Liquidity needs columns: datetime, close, volume")

    d = df.copy()
    d["datetime"] = to_utc_datetime(d["datetime"])
    d = d.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")

    price = pd.to_numeric(d["close"], errors="coerce")
    vol = pd.to_numeric(d["volume"], errors="coerce")

    # Dollar Volume
    qv_col = detect_quote_volume_column(d)
    if qv_col:
        dollar_vol = pd.to_numeric(d[qv_col], errors="coerce")
    else:
        dollar_vol = price * vol

    # True Range
    high = pd.to_numeric(d["high"], errors="coerce") if "high" in d.columns else None
    low = pd.to_numeric(d["low"], errors="coerce") if "low" in d.columns else None
    if high is not None and low is not None:
        prev_close = price.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
    else:
        tr = (price.pct_change().abs() * price).fillna(0)

    # 平滑
    dv_ewm = ewma(dollar_vol, span_hours)
    tr_ewm = ewma(tr.replace(0, np.nan).fillna(tr.median()), span_hours)

    lp_hr = (dv_ewm / tr_ewm).replace([np.inf, -np.inf], np.nan)
    zwin = max(int(24 * zwin_days), 24)
    liq_z = zscore_rolling(lp_hr, zwin).dropna()

    # 对齐 price
    price = price.reindex(liq_z.index, method="nearest")
    return liq_z, price


def draw_liquidity_curves(
    df_hourly: pd.DataFrame,
    symbol: str,
    out_png: Path,
    span_hours: int = 24,
    zwin_days: int = 90,
    start_dt: pd.Timestamp | None = None,
):
    """
    绘制两张叠放图：
    - 上：全样本周线（Price-left / Liquidity-right）
    - 下：最近90天日线（Price-left / Liquidity-right）
    右轴添加水平虚线：-2, -1, 0, 1, 2
    """
    d = df_hourly.copy()
    d["datetime"] = to_utc_datetime(d["datetime"])
    d = d.dropna(subset=["datetime"]).sort_values("datetime")
    if start_dt is not None:
        d = d[d["datetime"] >= start_dt]
    if d.empty:
        print(f"[!] No data for liquidity curves after {start_dt}")
        return

    # 计算小时级 Liquidity Z 与 Price
    liq_z, price = compute_liquidity_series(d, span_hours, zwin_days)
    if liq_z.empty:
        print("[!] Liquidity Z is empty after computation.")
        return

    # 周/日重采样
    weekly = pd.DataFrame(
        {"Liquidity": liq_z.resample("1W").mean(), "Price": price.resample("1W").last()}
    ).dropna()

    daily = pd.DataFrame(
        {"Liquidity": liq_z.resample("1D").mean(), "Price": price.resample("1D").last()}
    ).dropna()

    tail = daily.iloc[-90:]

    # 绘图
    fig = plt.figure(figsize=(12, 8))

    # Weekly
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(weekly.index, weekly["Price"], color="tab:red", label="Price", alpha=0.85)
    ax1.set_ylabel("Price", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1b = ax1.twinx()
    ax1b.plot(weekly.index, weekly["Liquidity"], color="tab:blue", label="Liquidity Z", alpha=0.95)
    ax1b.set_ylabel("Liquidity Z", color="tab:blue")
    ax1b.tick_params(axis="y", labelcolor="tab:blue")
    for y in [-2, -1, 0, 1, 2]:
        ax1b.axhline(y=y, linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_title(f"{symbol} Weekly Price vs Liquidity Z")
    ax1.legend(loc="upper left"); ax1b.legend(loc="upper right")

    # Daily (last 90 days)
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(tail.index, tail["Price"], color="tab:red", label="Price", alpha=0.85)
    ax2.set_ylabel("Price", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2b = ax2.twinx()
    ax2b.plot(tail.index, tail["Liquidity"], color="tab:blue", label="Liquidity Z", alpha=0.95)
    ax2b.set_ylabel("Liquidity Z", color="tab:blue")
    ax2b.tick_params(axis="y", labelcolor="tab:blue")
    for y in [-2, -1, 0, 1, 2]:
        ax2b.axhline(y=y, linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_title(f"{symbol} Daily Price vs Liquidity Z (Last 90 Days)")
    ax2.legend(loc="upper left"); ax2b.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved liquidity figure -> {out_png}")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV (columns: datetime/date/timestamp, close, volume, [open_interest])")
    # ---- chip args ----
    ap.add_argument("--window_strength", type=int, default=5)
    ap.add_argument("--window_zone", type=int, default=60)
    ap.add_argument("--bins_pct", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--half_life", type=float, default=10)
    ap.add_argument("--quantile", type=float, default=0.8)
    ap.add_argument("--ma_period", type=int, default=10)
    ap.add_argument("--smooth", type=int, default=3)
    # ---- liquidity args ----
    ap.add_argument("--liquidity_start", type=str, default=None,
                    help="Liquidity curve start date (YYYY-MM-DD), independent of chip start")
    ap.add_argument("--liquidity_span_hours", type=int, default=24)
    ap.add_argument("--liquidity_zwindow_days", type=int, default=90)
    ap.add_argument("--out_dir", type=str, default="docs")
    args = ap.parse_args()

    # 读取与规范列
    df = pd.read_csv(args.csv)
    df.columns = [c.lower().strip() for c in df.columns]
    rename_map = {}
    col_map = {
        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "close": ["close", "last"],
        "volume": ["volume", "volumeusd", "volume usdt", "quoteassetvolume"],  # 容错
        "datetime": ["datetime", "date", "timestamp"]
    }
    for target, cands in col_map.items():
        for c in cands:
            if c in df.columns:
                rename_map[c] = target
                break
    df = df.rename(columns=rename_map)

    # 解析时间
    if "datetime" not in df.columns:
        raise SystemExit("[X] Missing column: datetime/date/timestamp")
    df["datetime"] = to_utc_datetime(df["datetime"])

    # 仅“筹码分析”使用文件名中的 start（保持老版本行为）
    m = re.search(r'_(\d{4}-\d{2}-\d{2})_to_', args.csv)
    if m:
        chip_start_dt = pd.to_datetime(m.group(1), utc=True)
        df_chip = df[df["datetime"] >= chip_start_dt].copy()
        print(f"[*] Chip filtered from {chip_start_dt.date()}, rows={len(df_chip)}")
    else:
        df_chip = df.copy()

    # 必要列检查
    for col in ["close", "volume"]:
        if col not in df.columns:
            raise SystemExit(f"[X] Missing column: {col}")

    ensure_docs_dir()
    symbol = extract_symbol_from_filename(args.csv)

    # 统一的时间戳（保证本次筹码图 & 流动性图一致）
    ts = datetime.now().strftime("%Y%m%d_%H")

    # ========== 筹码分析 ==========
    chip_df = compute_chipzones_futures_logic(
        df=df_chip,
        window_zone=args.window_zone,
        bins_pct=args.bins_pct,
        beta=args.beta,
        half_life=args.half_life,
        quantile=args.quantile,
    )

    if "open_interest" in df_chip.columns:
        oi = pd.to_numeric(df_chip["open_interest"], errors="coerce").fillna(0.0)
        delta_oi = oi.diff().clip(lower=0.0).fillna(0.0).values
    else:
        delta_oi = np.zeros(len(df_chip), dtype=float)

    decay = exp_half_life_decay(df_chip["datetime"], half_life_days=float(args.half_life))
    eff_weight = df_chip["volume"].astype(float).values * np.power(1.0 + delta_oi, float(args.beta)) * decay

    long_curve, short_curve = compute_long_short(
        df_chip["close"], eff_weight, args.window_strength, args.smooth, args.window_zone
    )

    # 输出 CSV（不加时间戳，便于其他程序直接读取）
    chip_csv = Path(args.out_dir) / f"{symbol}_chip_strength.csv"
    chip_out = chip_df.copy()
    if "strength" not in chip_out.columns:
        chip_out["strength"] = chip_out["avg_strength"]
    chip_out["low"] = chip_out["low"].apply(fmt_price)
    chip_out["high"] = chip_out["high"].apply(fmt_price)
    chip_out.to_csv(chip_csv, index=False)
    print(f"[OK] Saved chip_strength table -> {chip_csv}")

    # 先删除旧筹码图，再保存新图（防缓存）
    delete_old_chip_images(symbol, args.out_dir)
    chip_png = Path(args.out_dir) / f"{symbol}_chip_timeline_pro_{ts}.png"
    draw_timeline(df_chip, chip_df, str(chip_png), args.ma_period, long_curve, short_curve)

    # ========== 流动性曲线 ==========
    if args.liquidity_start:
        liq_start = pd.to_datetime(args.liquidity_start, utc=True)

        # 删除旧流动性图
        delete_old_liquidity_images(symbol, args.out_dir)

        liq_png = Path(args.out_dir) / f"{symbol}_liquidity_{liq_start.strftime('%Y-%m-%d')}_{ts}.png"

        # 仅传入必要列，保证 DatetimeIndex 生效
        cols = ["datetime", "close", "volume"]
        if {"open", "high", "low"}.issubset(set(df.columns)):
            cols = ["datetime", "open", "high", "low", "close", "volume"]

        draw_liquidity_curves(
            df_hourly=df[cols].copy(),
            symbol=symbol,
            out_png=liq_png,
            span_hours=args.liquidity_span_hours,
            zwin_days=args.liquidity_zwindow_days,
            start_dt=liq_start
        )


if __name__ == "__main__":
    main()
