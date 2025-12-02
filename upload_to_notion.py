#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_to_notion.py (Chip + Liquidity Hybrid Display, CDN Version - Fixed)
--------------------------------------------------------------------
✔ 使用 jsDelivr CDN 显示图片/CSV（避免 raw/github.io 加载失败）
✔ 删除 ?v=hash / #v=hash（Notion 对 query/hash 链接不稳定）
✔ 递归扫描 docs/，支持带时间戳文件名
✔ 支持：
    - {SYMBOL}_chip_timeline_pro_YYYYMMDD_HH.png
    - {SYMBOL}_liquidity_*.png
    - {SYMBOL}_chip_strength.csv
✔ 每个 symbol 使用 heading_2，Notion 右侧自动生成目录导航
"""

import os
import sys
from datetime import datetime, timezone

import pandas as pd
from notion_client import Client

# ======== CONFIG ========
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
NOTION_SUMMARY_PAGE_ID = os.environ.get("NOTION_SUMMARY_PAGE_ID")

if not NOTION_TOKEN or not NOTION_SUMMARY_PAGE_ID:
    sys.exit("[X] Missing required Notion secrets. Please check GitHub Secrets configuration.")

notion = Client(auth=NOTION_TOKEN)

# ---- 固定使用 jsDelivr CDN（最稳定方式） ----
# 对应仓库路径：...@main/docs
CDN_BASE = "https://cdn.jsdelivr.net/gh/CMUJIN/crypto_csi_toolkit@main/docs"


# ======== HELPERS ========
def fmt_price(v: float) -> str:
    """格式化价格"""
    try:
        v = float(v)
        return f"{int(round(v))}" if v >= 100 else f"{v:.2f}"
    except Exception:
        return str(v)


def clear_summary_blocks():
    """安全清空汇总页内容"""
    print("[~] Clearing old summary blocks (safe mode)...")
    try:
        blocks = notion.blocks.children.list(NOTION_SUMMARY_PAGE_ID).get("results", [])
        removed = 0
        for blk in blocks:
            if blk.get("type") not in ("child_page", "child_database"):
                notion.blocks.delete(blk["id"])
                removed += 1
        print(f"[OK] Summary cleared safely: {removed} blocks removed.")
    except Exception as e:
        print(f"[!] Failed to clear summary safely: {e}")


def build_table_block(df: pd.DataFrame):
    """将 DataFrame 渲染成 Notion table block"""
    if df.empty:
        return []

    header = list(df.columns)
    header_row = {
        "object": "block",
        "type": "table_row",
        "table_row": {
            "cells": [[{"type": "text", "text": {"content": col}}] for col in header]
        },
    }

    rows = [header_row]
    for _, row in df.iterrows():
        cells = []
        for col in header:
            val = row[col]
            if isinstance(val, (int, float)) and pd.notna(val):
                if "low" in col.lower() or "high" in col.lower():
                    val = f"{val:.2f}" if val < 100 else f"{int(round(val))}"
                else:
                    val = f"{val:.3f}"
            else:
                val = str(val)
            cells.append([{"type": "text", "text": {"content": val}}])

        rows.append({
            "object": "block",
            "type": "table_row",
            "table_row": {"cells": cells}
        })

    return [{
        "object": "block",
        "type": "table",
        "table": {
            "table_width": len(header),
            "has_column_header": True,
            "has_row_header": False,
            "children": rows
        }
    }]


def walk_docs():
    """递归列出 docs/ 下的所有文件（含子目录）"""
    for root, _, files in os.walk("docs"):
        for file in files:
            yield os.path.join(root, file)


# ======== MAIN ========
def upload_to_notion():
    print("[*] Notion sync start (Chip + Liquidity via CDN)...")

    data_items = []

    # -------- 扫描所有 docs/* 文件 --------
    for full_path in walk_docs():
        if not full_path.lower().endswith(".png"):
            continue

        file = os.path.basename(full_path)
        rel_path = os.path.relpath(full_path, "docs").replace("\\", "/")

        # ================== 筹码时间线图 ==================
        # 兼容：SYMBOL_chip_timeline_pro_YYYYMMDD_HH.png
        if file.endswith(".png") and "_chip_timeline_pro_" in file:
            name = os.path.splitext(file)[0]
            parts = name.split("_")

            # 提取 symbol（优先带 USDT/USD 的那一段）
            symbol = next((p for p in parts if "USDT" in p.upper() or "USD" in p.upper()), parts[0])
            symbol = symbol.upper()

            # CDN 图片链接（支持子目录）
            chart_url = f"{CDN_BASE}/{rel_path}"

            # 对应 CSV：{SYMBOL}_chip_strength.csv，认为它和图片在同一目录
            csv_path_local = os.path.join(os.path.dirname(full_path), f"{symbol}_chip_strength.csv")
            csv_url = None
            if os.path.exists(csv_path_local):
                csv_rel = os.path.relpath(csv_path_local, "docs").replace("\\", "/")
                csv_url = f"{CDN_BASE}/{csv_rel}"

            data_items.append({
                "symbol": symbol,
                "chart_url": chart_url,
                "csv_path": csv_path_local if os.path.exists(csv_path_local) else None,
                "csv_url": csv_url,
                "type": "chip"
            })

        # ================== 流动性图 ==================
        # 兼容：SYMBOL_liquidity_YYYY-MM-DD_YYYYMMDD_HH.png 等形式
        elif file.endswith(".png") and "_liquidity_" in file:
            name = os.path.splitext(file)[0]
            parts = name.split("_")
            # 文件名形如：AAVEUSDT_liquidity_2024-10-01_20251202_03
            symbol = parts[0].upper()

            chart_url = f"{CDN_BASE}/{rel_path}"

            data_items.append({
                "symbol": symbol,
                "chart_url": chart_url,
                "csv_path": None,
                "csv_url": None,
                "type": "liquidity"
            })

    if not data_items:
        print("[!] No chip or liquidity analysis found in docs/.")
        return

    # -------- 清空 Notion 汇总页 --------
    clear_summary_blocks()

    # -------- 分 symbol 聚合 --------
    grouped = {}
    for item in data_items:
        grouped.setdefault(item["symbol"], []).append(item)

    children = []

    for symbol, items in grouped.items():
        # 每个 symbol 一个 heading_2：Notion 右侧自动生成目录导航
        children.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": symbol}}]
            }
        })

        for item in items:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            label = "📊 筹码分析" if item["type"] == "chip" else "💧 流动性曲线"

            # 更新说明
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {"type": "text", "text": {"content": f"{label}（更新于 {ts}）"}}
                    ]
                }
            })

            # 图片（CDN 外链）
            children.append({
                "object": "block",
                "type": "image",
                "image": {"type": "external", "external": {"url": item["chart_url"]}}
            })

            # 如果是筹码图，追加 CSV 表格
            if item["type"] == "chip" and item["csv_path"] and os.path.exists(item["csv_path"]):
                try:
                    df = pd.read_csv(item["csv_path"])
                    children.extend(build_table_block(df))
                except Exception as e:
                    children.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"type": "text", "text": {"content": f"[X] CSV load failed: {e}"}}
                            ]
                        }
                    })

    # 一次性写入 Notion
    notion.blocks.children.append(NOTION_SUMMARY_PAGE_ID, children=children)
    print(f"[OK] Summary updated (via CDN) with {len(grouped)} symbols.")


if __name__ == "__main__":
    upload_to_notion()
