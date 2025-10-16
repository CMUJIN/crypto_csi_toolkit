import os
import csv
import pandas as pd
from notion_client import Client
from datetime import datetime

# === 环境变量 ===
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
NOTION_SUMMARY_PAGE_ID = os.getenv("NOTION_SUMMARY_PAGE_ID")

notion = Client(auth=NOTION_TOKEN)


# === 自动检测数据库标题列 ===
def detect_title_property():
    db = notion.databases.retrieve(NOTION_DATABASE_ID)
    for k, v in db["properties"].items():
        if v["type"] == "title":
            print(f"[OK] Detected title property: {k}")
            return k
    return "Name"


# === 清空数据库中旧页面 ===
def clear_database(title_prop):
    print("[~] Clearing old database records...")
    pages = notion.databases.query(database_id=NOTION_DATABASE_ID).get("results", [])
    for page in pages:
        pid = page["id"]
        title = page["properties"][title_prop]["title"]
        name = title[0]["plain_text"] if title else "(untitled)"
        try:
            notion.pages.update(page_id=pid, archived=True)
            print(f"  - Deleted: {name}")
        except Exception as e:
            print(f"  [X] Failed to delete {name}: {e}")
    print("[OK] Database cleared.")


# === 读取Top20%筹码区间数据 ===
def read_top_chip_strength(csv_path, top_pct=0.2, max_rows=30):
    df = pd.read_csv(csv_path)
    if "strength" not in df.columns:
        print(f"[!] Missing 'strength' column in {csv_path}, using first {max_rows} rows.")
        return df.head(max_rows)

    df = df.sort_values(by="strength", ascending=False)
    cutoff = int(len(df) * top_pct)
    if cutoff < 1:
        cutoff = 1
    top_df = df.head(cutoff)
    if len(top_df) > max_rows:
        top_df = top_df.head(max_rows)
    return top_df


# === 清空主页汇总区块 ===
def clear_summary_blocks():
    try:
        children = notion.blocks.children.list(block_id=NOTION_SUMMARY_PAGE_ID).get("results", [])
        for block in children:
            bid = block["id"]
            notion.blocks.delete(block_id=bid)
        print(f"[OK] Cleared old blocks for summary page ({len(children)} removed).")
    except Exception as e:
        print(f"[X] Failed to clear summary page: {e}")


# === 更新数据库（重新插入最新分析结果） ===
def update_database_entry(symbol, csv_url, chart_url, title_prop):
    notion.pages.create(
        parent={"database_id": NOTION_DATABASE_ID},
        properties={
            title_prop: {"title": [{"text": {"content": symbol}}]},
            "Timeframe": {"rich_text": [{"text": {"content": "1h"}}]},
            "Updated": {"date": {"start": datetime.now().isoformat()}},
            "CSV": {"url": csv_url},
            "Chart": {"url": chart_url},
        },
    )
    print(f"[+] Inserted {symbol} into database.")


# === 更新主页汇总区 ===
def update_summary_page(data_items):
    clear_summary_blocks()
    children_blocks = []

    for item in data_items:
        symbol = item["symbol"]
        csv_path = item["csv_path"]
        chart_url = item["chart_url"]
        df = read_top_chip_strength(csv_path, top_pct=0.2, max_rows=30)

        header = list(df.columns)
        header_row = {
            "object": "block",
            "type": "table_row",
            "table_row": {"cells": [[{"type": "text", "text": {"content": h}}] for h in header]},
        }

        data_rows = [
            {
                "object": "block",
                "type": "table_row",
                "table_row": {"cells": [[{"type": "text", "text": {"content": str(v)}}] for v in row]},
            }
            for _, row in df.iterrows()
        ]

        table_block = {
            "object": "block",
            "type": "table",
            "table": {
                "table_width": len(header),
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

        children_blocks.extend([
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": f"{symbol} Chip Strength (Top 20%)"}}]},
            },
            image_block,
            table_block,
        ])

    notion.blocks.children.append(
        block_id=NOTION_SUMMARY_PAGE_ID,
        children=children_blocks,
    )
    print("[OK] Summary page updated successfully ✅")


# === 主程序 ===
def upload_to_notion():
    print("[*] Starting Notion sync...")
    title_prop = detect_title_property()
    clear_database(title_prop)

    data_items = []
    for filename in os.listdir("docs"):
        if filename.endswith("_chip_strength.csv"):
            symbol = filename.split("_")[1]
            base = filename.replace("_chip_strength.csv", "")
            csv_path = f"docs/{filename}"
            csv_url = f"https://cmujin.github.io/crypto_csi_toolkit/{filename}"
            chart_url = f"https://cmujin.github.io/crypto_csi_toolkit/{base}_chip_timeline_pro.png"
            data_items.append({"symbol": symbol, "csv_url": csv_url, "chart_url": chart_url, "csv_path": csv_path})

    print(f"[*] Found {len(data_items)} chip_strength CSVs to upload.")
    for item in data_items:
        update_database_entry(item["symbol"], item["csv_url"], item["chart_url"], title_prop)

    update_summary_page(data_items)
    print("[OK] All updates complete ✅")


if __name__ == "__main__":
    upload_to_notion()
