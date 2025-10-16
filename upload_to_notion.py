import os
import csv
from notion_client import Client
from datetime import datetime

# === 环境变量 ===
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
NOTION_SUMMARY_PAGE_ID = os.getenv("NOTION_SUMMARY_PAGE_ID")

notion = Client(auth=NOTION_TOKEN)


# === 自动检测数据库标题列名 ===
def detect_title_property():
    try:
        db = notion.databases.retrieve(NOTION_DATABASE_ID)
        for key, val in db["properties"].items():
            if val["type"] == "title":
                print(f"[OK] Detected title property: {key}")
                return key
    except Exception as e:
        print(f"[X] Failed to detect title property: {e}")
    return "Name"  # fallback


# === 读取 CSV 的前若干行预览 ===
def read_csv_preview(csv_path, max_rows=30):
    rows = []
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)
    return header, rows


# === 更新数据库中的记录 ===
def update_database_entry(symbol, csv_url, chart_url, title_prop):
    now = datetime.now().strftime("%B %d, %Y %I:%M %p")

    try:
        pages = notion.databases.query(
            **{
                "database_id": NOTION_DATABASE_ID,
                "filter": {"property": title_prop, "title": {"equals": symbol}},
            }
        )
    except Exception as e:
        print(f"[X] Database query failed: {e}")
        return

    if pages.get("results"):
        page_id = pages["results"][0]["id"]
        print(f"[~] Updated {symbol} in database ({now})")
        notion.pages.update(
            page_id=page_id,
            properties={
                "Timeframe": {"rich_text": [{"text": {"content": "1h"}}]},
                "Updated": {"date": {"start": datetime.now().isoformat()}},
                "CSV": {"url": csv_url},
                "Chart": {"url": chart_url},
            },
        )
    else:
        print(f"[+] Creating new entry for {symbol}")
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


# === 更新汇总页面 ===
def update_summary_page(data_items):
    if not NOTION_SUMMARY_PAGE_ID:
        print("[!] No summary page ID provided. Skip summary update.")
        return

    print("[~] Appending updated summary safely (no clearing)...")

    children_blocks = []
    for item in data_items:
        symbol = item["symbol"]
        csv_url = item["csv_url"]
        chart_url = item["chart_url"]
        csv_path = item["csv_path"]

        header, preview = read_csv_preview(csv_path, max_rows=30)

        # 表头
        header_row = {
            "object": "block",
            "type": "table_row",
            "table_row": {"cells": [[{"type": "text", "text": {"content": h}}] for h in header]},
        }

        # 数据行
        data_rows = [
            {
                "object": "block",
                "type": "table_row",
                "table_row": {"cells": [[{"type": "text", "text": {"content": str(v)}}] for v in row]},
            }
            for row in preview
        ]

        # 生成表格 + 图片 + 标题块
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

        children_blocks.extend(
            [
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"type": "text", "text": {"content": f"{symbol} Analysis"}}]},
                },
                image_block,
                table_block,
            ]
        )

    # ✅ 正确调用 Notion API append
    notion.blocks.children.append(
        block_id=NOTION_SUMMARY_PAGE_ID,
        children=children_blocks,
    )

    print("[OK] Summary page updated safely ✅")


# === 主执行逻辑 ===
def upload_to_notion():
    print("[*] Starting Notion sync...")

    title_prop = detect_title_property()
    data_items = []

    for filename in os.listdir("docs"):
        if filename.endswith("_chip_timeline_pro.png"):
            symbol = filename.split("_")[1]
            base = filename.replace("_chip_timeline_pro.png", "")
            csv_path = f"docs/{base}.csv"
            if not os.path.exists(csv_path):
                continue
            csv_url = f"https://cmujin.github.io/crypto_csi_toolkit/{base}.csv"
            chart_url = f"https://cmujin.github.io/crypto_csi_toolkit/{filename}"
            data_items.append({"symbol": symbol, "csv_url": csv_url, "chart_url": chart_url, "csv_path": csv_path})

    print(f"[*] Found {len(data_items)} CSV files to sync.")
    for item in data_items:
        update_database_entry(item["symbol"], item["csv_url"], item["chart_url"], title_prop)

    update_summary_page(data_items)


# === 程序入口 ===
if __name__ == "__main__":
    upload_to_notion()
