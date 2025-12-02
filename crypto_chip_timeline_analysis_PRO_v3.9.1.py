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
        if (ra >= thr_r_
