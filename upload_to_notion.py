#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_to_notion.py (Chip + Liquidity Hybrid Display, CDN Version)
--------------------------------------------------------------------
✔ 使用 jsDelivr CDN 显示图片/CSV（避免 raw/github.io 加载失败）
✔ 删除 ?v=hash / #v=hash（Notion 对 query/hash 链接不稳定）
✔ 保留全部原有业务逻辑、表格生成、symbol 提取、分组展示
"""

import os, sys, pandas as pd
from datetime import datetime, timezone
from notion_client import Client

# ======== CONFIG ========
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
NOTION_SUMMARY_PAGE_ID = os.environ.get("NOTION_SUMMARY_PAGE_ID")

if not NOTION_TOKEN or not NOTION_SUMMARY_PAGE_ID:
    sys.exit("[X] Missing required Notion secrets. Please check GitHub Secrets configuration.")

notion = Client(auth=NOTION_TOKEN)

# ---- 固定使用 jsDelivr CDN（最稳定方式） ----
CDN_BASE = "https://cdn.jsdelivr.net/gh/CMUJIN/crypto_csi_toolkit@main/docs"


# ======== HELPERS ========
def fmt_price(v: float) -> str:
    """格式化价格"""
    try:
        v = float(v)
        return f"{int(round(v))}" if v >= 100 else f"{v:.2f}"
    except:
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
    """表格渲染"""
    if df.empty:
        return []

    header = list(df.columns)
    header_row = {
        "object": "block",
        "type": "table_row",
        "table_row": {"cells": [[{"type": "text", "text": {"content": col}}] for col in header]},
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


# ======== MAIN ========
def upload_to_notion():
    print("[*] Notion sync start (Chip + Liquidity via CDN)...")

    data_items = []

    for file in os.listdir("docs"):
        # 筹码分析
        if file.endswith("_chip_timeline_pro.png"):
            name = os.path.splitext(file)[0]
            parts = name.split("_")
            symbol = next((p for p in parts if "USDT" in p or "USD" in p), parts[0])

            # ---- CDN 链接（无 hash/query）----
            chart_url = f"{CDN_BASE}/{file}"

            csv_path_local = os.path.join("docs", file.replace("_chip_timeline_pro.png", "_chip_strength.csv"))
            csv_file = os.path.basename(csv_path_local)
            csv_url = f"{CDN_BASE}/{csv_file}"

            if os.path.exists(csv_path_local):
                data_items.append({
                    "symbol": symbol,
                    "chart_url": chart_url,
                    "csv_path": csv_path_local,
                    "csv_url": csv_url,
                    "type": "chip"
                })

        # 流动性曲线
        elif "_liquidity_" in file and file.endswith(".png"):
            symbol = file.split("_")[0]
            chart_url = f"{CDN_BASE}/{file}"

            data_items.append({
                "symbol": symbol,
                "chart_url": chart_url,
                "csv_path": None,
                "csv_url": None,
                "type": "liquidity"
            })

    if not data_items:
        print("[!] No chip or liquidity analysis found.")
        return

    clear_summary_blocks()
    children = []

    grouped = {}
    for item in data_items:
        grouped.setdefault(item["symbol"], []).append(item)

    for symbol, items in grouped.items():

        children.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": symbol}}]}
        })

        for item in items:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            label = "📊 筹码分析" if item["type"] == "chip" else "💧 流动性曲线"

            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": f"{label}（更新于 {ts}）"}}]}
            })

            # ---- 图片使用 CDN ----
            children.append({
                "object": "block",
                "type": "image",
                "image": {"type": "external", "external": {"url": item["chart_url"]}}
            })

            # ---- 追加筹码 CSV 表格 ----
            if item["type"] == "chip" and item["csv_path"] and os.path.exists(item["csv_path"]):
                try:
                    df = pd.read_csv(item["csv_path"])
                    children.extend(build_table_block(df))
                except Exception as e:
                    children.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"type": "text", "text": {"content": f"[X] CSV load failed: {e}"}}]}
                    })

    notion.blocks.children.append(NOTION_SUMMARY_PAGE_ID, children=children)
    print(f"[OK] Summary updated (via CDN) with {len(grouped)} symbols.")


if __name__ == "__main__":
    upload_to_notion()
