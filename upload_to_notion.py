import os
import csv
from notion_client import Client
from datetime import datetime

# === 环境变量 ===
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
NOTION_SUMMARY_PAGE_ID = os.getenv("NOTION_SUMMARY_PAGE_ID")

notion = Client(auth=NOTION_TOKEN)

# === 读取本地 CSV 数据 ===
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

# === 上传到 Notion 数据库 ===
def update_database_entry(symbol, csv_url, chart_url):
    now = datetime.now().strftime("%B %d, %Y %I:%M %p")

    # 查询是否已存在该 symbol
    pages = notion.databases.query(
        **{
            "database_id": NOTION_DATABASE_ID,
            "filter": {"property": "Name", "title": {"equals": symbol}},
        }
    )

    if pages["results"]:
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
                "Name": {"title": [{"text": {"content": symbol}}]},
                "Timeframe": {"rich_text": [{"text": {"content": "1h"}}]},
                "Updated": {"date": {"start": datetime.now().isoformat()}},
                "CSV": {"url": csv_url},
                "Chart": {"url": chart_url},
            },
        )

# === 更新主页面汇总展示 ===
def update_summary_page(data_items):
    if not NOTION_SUMMARY_PAGE_ID:
        print("[!] No summary page ID provided. Skip summary update.")
        return

    print("[~] Appending updated summary without clearing old blocks...")

    children_blocks = []
    for item in data_items:
        symbol = item["symbol"]
        csv_url = item["csv_url"]
        chart_url = item["chart_url"]
        csv_path = item["csv_path"]

        header, preview = read_csv_preview(csv_path, max_rows=30)

        # 构造 CSV 表格块
        table_block = {
            "object": "block",
            "type": "table",
            "table": {
                "table_width": len(header),
                "has_column_header": True,
                "has_row_header": False,
                "children": [
                    {
                        "object": "block",
                        "type": "table_row",
                        "table_row": {"cells": [[{"type": "text", "text": {"content": h}}] for h in header]},
                    }
                ]
                + [
                    {
                        "object": "block",
                        "type": "table_row",
                        "table_row": {"cells": [[{"type": "text", "text": {"content": str(v)}}] for v in row]},
                    }
                    for row in preview
                ],
            },
        }

        # 构造图片块
        image_block = {
            "object": "block",
            "type": "image",
            "image": {"type": "external", "external": {"url": chart_url}},
        }

        children_blocks.append(
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": f"{symbol} Analysis"}}]},
            }
        )
        children_blocks.append(image_block)
        children_blocks.append(table_block)

    # ✅ 安全模式：仅追加，不清空原有内容
    notion.blocks.children.append(NOTION_SUMMARY_PAGE_ID, {"children": children_blocks})
    print("[OK] Summary page updated safely ✅")

# === 主执行逻辑 ===
def upload_to_notion():
    print("[*] Starting Notion sync...")

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
        update_database_entry(item["symbol"], item["csv_url"], item["chart_url"])

    # 🔐 防止页面被清空
    update_summary_page(data_items)

# === 程序入口 ===
if __name__ == "__main__":
    upload_to_notion()
