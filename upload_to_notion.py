#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_to_notion.py (Display-Only Mode)
----------------------------------------
✅ 从 Secrets 读取所有 Notion ID
✅ 安全清理 Summary 页面内容
✅ 仅生成图表 + 完整字段表格
✅ 每个品种前显示最新更新时间
❌ 不写入数据库（仅展示）
"""

import os, sys, pandas as pd
from datetime import datetime, timezone
from notion_client import Client

# ======== CONFIG ========
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID")
NOTION_SUMMARY_PAGE_ID = os.environ.get("NOTION_SUMMARY_PAGE_ID")

if not NOTION_TOKEN or not NOTION_SUMMARY_PAGE_ID:
    sys.exit("[X] Missing required Notion secrets. Please check GitHub Secrets configuration.")

notion = Client(auth=NOTION_TOKEN)


# ======== HELPERS ========
def fmt_price(v: float) -> str:
    """格式化价格：大于100取整，否则两位小数"""
    try:
        v = float(v)
        return f"{int(round(v))}" if v >= 100 else f"{v:.2f}"
    except:
        return str(v)


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


def build_table_block(df: pd.DataFrame):
    """生成完整字段表格，与数据库 CSV 一致"""
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
    return table_block


# ======== CORE ========
def update_summary(chip_data):
    """重建汇总页（显示图表 + 更新时间 + 表格）"""
    clear_summary_blocks()
    children = []

    for item in chip_data:
        symbol, chart_url, chip_csv = item["symbol"], item["chart_url"], item["csv_path"]

        # 更新时间
        ts = os.path.getmtime(chip_csv)
        ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # 标题
        children.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": symbol}}]}
        })

        # 更新时间标识
        children.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": f"📅 数据更新时间：{ts_str}"}}]
            }
        })

        # 图片
        children.append({
            "object": "block",
            "type": "image",
            "image": {"type": "external", "external": {"url": chart_url}}
        })

        # 表格（完整 chip_strength）
        try:
            df = pd.read_csv(chip_csv)
            children.extend(build_table_block(df))
        except Exception as e:
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": f"[X] Failed to load table: {e}"}}]
                }
            })

    notion.blocks.children.append(NOTION_SUMMARY_PAGE_ID, children=children)
    print(f"[OK] Summary updated successfully with {len(chip_data)} items.")


# ======== MAIN ========
def upload_to_notion():
    print("[*] Notion sync start (Display-only mode)...")
    data_items = []

    for file in os.listdir("docs"):
        if file.endswith("_chip_timeline_pro.png"):
            symbol = file.split("_")[1]
            version_tag = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            chart_url = f"https://cmujin.github.io/crypto_csi_toolkit/{file}#v={version_tag}"
            csv_path = os.path.join("docs", file.replace("_chip_timeline_pro.png", "_chip_strength.csv"))
            csv_url = f"https://cmujin.github.io/crypto_csi_toolkit/{os.path.basename(csv_path)}"
            if os.path.exists(csv_path):
                data_items.append({"symbol": symbol, "chart_url": chart_url, "csv_path": csv_path, "csv_url": csv_url})

    if not data_items:
        print("[!] No data found, abort.")
        return

    print(f"[OK] Found {len(data_items)} chip_strength CSVs.")
    # ❌ 不再写入数据库
    # archive_old_records()
    # for item in data_items:
    #     create_or_update_record(item["symbol"], item["chart_url"], item["csv_url"])

    update_summary(data_items)
    print("[OK] Summary page updated successfully (Display-only mode)!")

if __name__ == "__main__":
    upload_to_notion()
