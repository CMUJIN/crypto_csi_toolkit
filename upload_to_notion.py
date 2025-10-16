import os
import csv
import math
import pandas as pd
from notion_client import Client
from datetime import datetime

# ===== Env =====
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")          # 数据库本体（保留结构，仅清空记录）
NOTION_SUMMARY_PAGE_ID = os.getenv("NOTION_SUMMARY_PAGE_ID")  # 主页，用于展示图+表（只重建我们自己的容器块）

notion = Client(auth=NOTION_TOKEN)

# ===== Utils =====
def detect_title_property():
    """自动识别数据库标题列（title 类型）。"""
    db = notion.databases.retrieve(NOTION_DATABASE_ID)
    for k, v in db["properties"].items():
        if v.get("type") == "title":
            print(f"[OK] Detected title property: {k}")
            return k
    return "Name"

def fmt_price(v):
    """价格格式：>=100 取整数；<100 保留两位小数；非数值原样返回。"""
    try:
        x = float(v)
    except Exception:
        return str(v)
    if math.isnan(x) or math.isinf(x):
        return str(v)
    if x >= 100:
        return f"{int(round(x))}"
    return f"{x:.2f}"

def read_top_chip_strength(csv_path, top_pct=0.2, max_rows=30):
    """读取 *_chip_strength.csv，按 strength 取前 20%，最多 30 行。"""
    df = pd.read_csv(csv_path)
    if "strength" not in df.columns:
        print(f"[!] Missing 'strength' column in {csv_path}, fallback head({max_rows}).")
        return df.head(max_rows)

    df = df.sort_values(by="strength", ascending=False)
    cutoff = max(1, int(len(df) * top_pct))
    top_df = df.head(cutoff)
    if len(top_df) > max_rows:
        top_df = top_df.head(max_rows)
    return top_df

# ===== Database Ops (records only) =====
def clear_database_records(title_prop):
    """归档数据库中所有现有页面记录（不删除数据库本体）。"""
    print("[~] Clearing old database records (archive pages only)...")
    next_cursor = None
    total = 0
    while True:
        resp = notion.databases.query(
            database_id=NOTION_DATABASE_ID,
            start_cursor=next_cursor
        )
        results = resp.get("results", [])
        for page in results:
            pid = page["id"]
            title = page["properties"].get(title_prop, {}).get("title", [])
            name = title[0]["plain_text"] if title else "(untitled)"
            try:
                notion.pages.update(page_id=pid, archived=True)
                total += 1
                print(f"  - Archived record: {name}")
            except Exception as e:
                print(f"  [X] Failed to archive {name}: {e}")
        if not resp.get("has_more"):
            break
        next_cursor = resp.get("next_cursor")
    print(f"[OK] Database records cleared: {total}")

def insert_database_record(symbol, csv_url, chart_url, title_prop, timeframe="1h"):
    """插入单条最新记录。"""
    notion.pages.create(
        parent={"database_id": NOTION_DATABASE_ID},
        properties={
            title_prop: {"title": [{"text": {"content": symbol}}]},
            "Timeframe": {"rich_text": [{"text": {"content": timeframe}}]},
            "Updated": {"date": {"start": datetime.now().isoformat()}},
            "CSV": {"url": csv_url},
            "Chart": {"url": chart_url},
        },
    )
    print(f"[+] Inserted record: {symbol}")

# ===== Summary Page Ops (safe container) =====
CONTAINER_TITLE = "Auto Report"

def delete_old_container():
    """仅删除我们脚本创建的容器块（callout 标题为 CONTAINER_TITLE），不动其它块/数据库视图。"""
    try:
        children = notion.blocks.children.list(block_id=NOTION_SUMMARY_PAGE_ID).get("results", [])
        removed = 0
        for block in children:
            if block["type"] == "callout":
                rich = block["callout"].get("rich_text", [])
                title_text = "".join([t["plain_text"] for t in rich if t.get("plain_text")])
                if title_text.strip() == CONTAINER_TITLE:
                    notion.blocks.delete(block_id=block["id"])
                    removed += 1
        print(f"[OK] Removed old container blocks: {removed}")
    except Exception as e:
        print(f"[X] Failed to cleanup old container: {e}")

def build_symbol_section_blocks(symbol, csv_path, chart_url):
    """构建单个币种的图 + Top20%表格（对 Low/High 做定制格式化）。"""
    df = read_top_chip_strength(csv_path, top_pct=0.2, max_rows=30)

    # 表头
    headers = list(df.columns)
    header_row = {
        "object": "block",
        "type": "table_row",
        "table_row": {
            "cells": [[{"type": "text", "text": {"content": h}}] for h in headers]
        },
    }

    # 行数据（Low/High 格式处理）
    low_keys = {c for c in df.columns if c.lower() == "low"}
    high_keys = {c for c in df.columns if c.lower() == "high"}

    data_rows = []
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            val = row[col]
            if col in low_keys or col in high_keys:
                cells.append([{"type": "text", "text": {"content": fmt_price(val)}}])
            else:
                cells.append([{"type": "text", "text": {"content": str(val)}}])
        data_rows.append({
            "object": "block",
            "type": "table_row",
            "table_row": {"cells": cells},
        })

    table_block = {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": len(headers),
            "has_column_header": True,
            "has_row_header": False,
            "children": [header_row] + data_rows,
        },
    }

    image_block = {
        "object": "block",
        "type": "image",
        "image": {"type": "external", "external": {"url": chart_url}},
    }

    heading_block = {
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": [{"type": "text", "text": {"content": f"{symbol} Chip Strength (Top 20%)"}}]},
    }

    return [heading_block, image_block, table_block]

def rebuild_summary_container(sections_blocks):
    """新建容器块（callout），把所有币种的块作为 children 放进去。"""
    container = {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [{"type": "text", "text": {"content": CONTAINER_TITLE}}],
            # 不设置 icon，避免表情导致兼容问题
            "children": sections_blocks
        },
    }
    notion.blocks.children.append(
        block_id=NOTION_SUMMARY_PAGE_ID,
        children=[container],
    )
    print("[OK] Summary page rebuilt.")

# ===== Main =====
def upload_to_notion():
    print("[*] Notion sync start...")
    title_prop = detect_title_property()

    # 收集 chip_strength 结果（而非原始 CSV）
    items = []
    for fn in os.listdir("docs"):
        if fn.endswith("_chip_strength.csv"):
            base = fn.replace("_chip_strength.csv", "")
            parts = base.split("_")
            if len(parts) < 3:
                continue
            symbol = parts[1]
            csv_path = os.path.join("docs", fn)
            csv_url = f"https://cmujin.github.io/crypto_csi_toolkit/{fn}"
            chart_url = f"https://cmujin.github.io/crypto_csi_toolkit/{base}_chip_timeline_pro.png"
            items.append({"symbol": symbol, "csv_path": csv_path, "csv_url": csv_url, "chart_url": chart_url})

    print(f"[*] Found {len(items)} chip_strength CSVs.")

    # 1) 清空数据库记录（仅归档页面，不删库）
    clear_database_records(title_prop)

    # 2) 重新插入最新记录
    for it in items:
        insert_database_record(it["symbol"], it["csv_url"], it["chart_url"], title_prop)

    # 3) 重建主页报告容器（只删我们自己的容器，不动其它块/数据库视图）
    delete_old_container()

    sections = []
    for it in items:
        sections.extend(build_symbol_section_blocks(it["symbol"], it["csv_path"], it["chart_url"]))

    rebuild_summary_container(sections)
    print("[OK] Notion sync done.")

if __name__ == "__main__":
    upload_to_notion()
