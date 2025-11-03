#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_to_notion.py (Chip + Liquidity Display, No Liquidity CSV)
---------------------------------------------------------------
✅ 同步上传两类图表（Chip + Liquidity）
✅ 显示最新更新时间
✅ 仅显示 Chip Strength 表格
❌ 不显示 Liquidity CSV
❌ 不写入数据库
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


# ======== HELPERS ========
def clear_summary_blocks():
    """安全清空汇总页内容，仅删除普通块，不删除数据库或子页面"""
    print("[~] Clearing old summary blocks (safe mode)...")
    try:
        blocks = notion.blocks.children.list(NOTION_SUMMARY_PAGE_ID).get("results", [])
        removed = 0
        for blk in blocks:
            blk_type = blk.get("type")
            if blk_type not in ("child_database", "child_page"):
                notion.blocks.delete(blk["id"])
                removed += 1
        print(f"[OK] Summary cleared safely: {removed} blocks removed (database retained)")
    except Exception as e:
        print(f"[!] Failed to clear summary safely: {e}")


def build_table_block(df: pd.DataFrame, title: str = None, max_rows: int = 20):
    """生成 Notion 表格块，支持可选标题"""
    if df.empty:
        return []

    df = df.head(max_rows)
    header = list(df.columns)
    blocks = []

    if title:
        blocks.append({
            "object": "block",
            "type": "heading_3",
            "heading_3": {"rich_text": [{"type": "text", "text": {"content": title}}]}
        })

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
                val = f"{val:.4g}"
            else:
                val = str(val)
            cells.append([{"type": "text", "text": {"content": val}}])
        rows.append({"object": "block", "type": "table_row", "table_row": {"cells": cells}})

    table_block = [{
        "object": "block",
        "type": "table",
        "table": {
            "table_width": len(header),
            "has_column_header": True,
            "has_row_header": False,
            "children": rows
        }
    }]
    blocks.extend(table_block)
    return blocks


# ======== CORE ========
def update_summary(data_items):
    """重建汇总页（Chip + Liquidity 图像，但不显示 Liquidity CSV）"""
    clear_summary_blocks()
    children = []

    for item in data_items:
        symbol = item["symbol"]
        chip_chart_url = item.get("chip_chart_url")
        chip_csv = item.get("chip_csv")
        liq_chart_url = item.get("liq_chart_url")

        # 更新时间（取最新修改文件）
        ts_files = [f for f in [chip_csv] if f and os.path.exists(f)]
        ts = max([os.path.getmtime(f) for f in ts_files]) if ts_files else None
        ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts else "N/A"

        # ---- 币种标题 ----
        children.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": symbol}}]}
        })
        # 更新时间
        children.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": f"📅 数据更新时间：{ts_str}"}}]}
        })

        # ---- 筹码强度图 ----
        if chip_chart_url:
            children.append({
                "object": "block",
                "type": "image",
                "image": {"type": "external", "external": {"url": chip_chart_url}}
            })

        # ---- 流动性曲线图 ----
        if liq_chart_url:
            children.append({
                "object": "block",
                "type": "image",
                "image": {"type": "external", "external": {"url": liq_chart_url}}
            })

        # ---- 筹码表格 ----
        if chip_csv and os.path.exists(chip_csv):
            try:
                df_chip = pd.read_csv(chip_csv)
                children.extend(build_table_block(df_chip, title="Chip Strength Table"))
            except Exception as e:
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": f"[X] Failed to load chip CSV: {e}"}}]}
                })

    notion.blocks.children.append(NOTION_SUMMARY_PAGE_ID, children=children)
    print(f"[OK] Summary updated successfully with {len(data_items)} symbols.")


# ======== MAIN ========
def upload_to_notion():
    print("[*] Notion sync start (Chip + Liquidity, No CSV for Liquidity)...")
    data_items = []
    docs_path = "docs"
    base_url = "https://cmujin.github.io/crypto_csi_toolkit"

    for file in os.listdir(docs_path):
        if file.endswith("_chip_timeline_pro.png"):
            symbol = file.split("_")[1]
            version_tag = datetime.utcnow().strftime("%Y%m%d%H%M%S")

            chip_png = os.path.join(docs_path, file)
            chip_csv = os.path.join(docs_path, file.replace("_chip_timeline_pro.png", "_chip_strength.csv"))
            liq_png = os.path.join(docs_path, f"{symbol}_liquidity_curves.png")

            chip_chart_url = f"{base_url}/{file}#v={version_tag}"
            liq_chart_url = f"{base_url}/{os.path.basename(liq_png)}#v={version_tag}" if os.path.exists(liq_png) else None

            data_items.append({
                "symbol": symbol,
                "chip_chart_url": chip_chart_url,
                "chip_csv": chip_csv,
                "liq_chart_url": liq_chart_url,
            })

    if not data_items:
        print("[!] No chip analysis found, abort.")
        return

    print(f"[OK] Found {len(data_items)} chip items (with optional liquidity).")
    update_summary(data_items)
    print("[OK] Summary page updated successfully (Chip + Liquidity, No CSV for Liquidity)!")

if __name__ == "__main__":
    upload_to_notion()
