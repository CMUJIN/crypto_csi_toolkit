#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_to_notion.py (Chip + Liquidity Hybrid Display)
--------------------------------------------------------
✅ 从 Secrets 读取 Notion Token 与页面 ID
✅ 自动识别 docs/ 下的筹码分析图与流动性曲线图
✅ 筹码分析显示表格与更新时间
✅ 流动性曲线仅显示图片与更新时间
✅ 安全清理 Summary 页面，仅删除普通块，不删除数据库或子页面
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
    print("[*] Notion sync start (Chip + Liquidity, No CSV for Liquidity)...")
    data_items = []

    # 遍历 docs 目录，识别两类图表
    for file in os.listdir("docs"):
        # 筹码分析图
        if file.endswith("_chip_timeline_pro.png"):
            symbol = file.split("_")[1]
            version_tag = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            chart_url = f"https://cmujin.github.io/crypto_csi_toolkit/{file}#v={version_tag}"
            csv_path = os.path.join("docs", file.replace("_chip_timeline_pro.png", "_chip_strength.csv"))
            csv_url = f"https://cmujin.github.io/crypto_csi_toolkit/{os.path.basename(csv_path)}"
            if os.path.exists(csv_path):
                data_items.append({
                    "symbol": symbol,
                    "chart_url": chart_url,
                    "csv_path": csv_path,
                    "csv_url": csv_url,
                    "type": "chip"
                })

        # 流动性曲线图
        elif "_liquidity_" in file and file.endswith(".png"):
            symbol = file.split("_")[0]
            version_tag = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            chart_url = f"https://cmujin.github.io/crypto_csi_toolkit/{file}#v={version_tag}"
            data_items.append({
                "symbol": symbol,
                "chart_url": chart_url,
                "csv_path": None,
                "csv_url": None,
                "type": "liquidity"
            })

    if not data_items:
        print("[!] No chip or liquidity analysis found, abort.")
        return

    clear_summary_blocks()
    children = []

    # 分组显示：每个 symbol 汇总其两张图
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

            # 图片标题
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": f"{label}（更新于 {ts}）"}}]
                }
            })

            # 插入图像
            children.append({
                "object": "block",
                "type": "image",
                "image": {"type": "external", "external": {"url": item["chart_url"]}}
            })

            # 如果是筹码分析，追加表格
            if item["type"] == "chip" and item["csv_path"] and os.path.exists(item["csv_path"]):
                try:
                    df = pd.read_csv(item["csv_path"])
                    children.extend(build_table_block(df))
                except Exception as e:
                    children.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": f"[X] Failed to load table: {e}"}}]
                        }
                    })

    # 上传到 Notion 页面
    notion.blocks.children.append(NOTION_SUMMARY_PAGE_ID, children=children)
    print(f"[OK] Summary updated successfully with {len(grouped)} symbols (chip + liquidity).")


if __name__ == "__main__":
    upload_to_notion()
