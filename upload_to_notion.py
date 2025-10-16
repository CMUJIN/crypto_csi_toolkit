import os
import csv
from notion_client import Client
from datetime import datetime
from pathlib import Path

# 环境变量
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")   # 数据库
NOTION_SUMMARY_PAGE_ID = os.getenv("NOTION_SUMMARY_PAGE_ID")  # 主展示页

notion = Client(auth=NOTION_TOKEN)
REPO_BASE = "https://cmujin.github.io/crypto_csi_toolkit"


# ==== 工具函数 ====

def notion_url(path: str) -> str:
    ts = int(datetime.now().timestamp())
    return f"{REPO_BASE}/{path}?ver={ts}"


def parse_csv_preview(csv_path: Path, max_rows=30):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        headers = reader[0]
        rows = reader[1:max_rows + 1]
        if len(reader) > max_rows + 1:
            rows.append(["..."] * len(headers))
    return headers, rows


# ==== Notion数据库更新 ====

def update_database(symbol, freq, csv_url, png_url):
    query = notion.databases.query(
        database_id=NOTION_DATABASE_ID,
        filter={"property": "Name", "title": {"equals": symbol}},
    )
    updated_time = datetime.now().strftime("%B %d, %Y %I:%M %p")

    props = {
        "Name": {"title": [{"text": {"content": symbol}}]},
        "Timeframe": {"rich_text": [{"text": {"content": freq}}]},
        "Updated": {"date": {"start": datetime.now().astimezone().isoformat()}},
        "CSV": {"url": csv_url},
        "Chart": {"url": png_url},
    }

    if query["results"]:
        page_id = query["results"][0]["id"]
        notion.pages.update(page_id=page_id, properties=props)
        print(f"[~] Updated {symbol} in database ({updated_time})")
    else:
        notion.pages.create(parent={"database_id": NOTION_DATABASE_ID}, properties=props)
        print(f"[+] Created new database entry for {symbol}")


# ==== 主页面展示区块 ====

def build_symbol_section(symbol, png_url, csv_path):
    headers, rows = parse_csv_preview(csv_path)
    table_rows = []

    # Header
    table_rows.append({
        "object": "block",
        "type": "table_row",
        "table_row": {
            "cells": [[{"type": "text", "text": {"content": h}}] for h in headers]
        },
    })

    # Rows
    for row in rows:
        table_rows.append({
            "object": "block",
            "type": "table_row",
            "table_row": {
                "cells": [[{"type": "text", "text": {"content": str(c)}}] for c in row]
            },
        })

    # 返回区块
    return [
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": f"{symbol} Analysis"}}]
            },
        },
        {
            "object": "block",
            "type": "image",
            "image": {"type": "external", "external": {"url": png_url}},
        },
        {
            "object": "block",
            "type": "table",
            "table": {
                "has_column_header": True,
                "has_row_header": False,
                "table_width": len(headers),
                "children": table_rows,
            },
        },
        {"object": "block", "type": "divider", "divider": {}},
    ]


# ==== 主流程 ====

def upload_to_notion():
    docs_dir = Path("docs")
    csv_files = list(docs_dir.glob("Binance_*_to_latest.csv"))
    all_blocks = []

    print(f"[*] Found {len(csv_files)} CSV files to sync.")

    for csv_file in csv_files:
        symbol = csv_file.name.split("_")[1]
        png_file = csv_file.with_name(csv_file.stem + "_chip_timeline_pro.png")
        freq = csv_file.name.split("_")[2] if len(csv_file.name.split("_")) > 2 else "1h"

        csv_url = notion_url(csv_file.name)
        png_url = notion_url(png_file.name)

        # 1️⃣ 更新数据库
        update_database(symbol, freq, csv_url, png_url)

        # 2️⃣ 生成汇总展示块
        section = build_symbol_section(symbol, png_url, csv_file)
        all_blocks.extend(section)

    # 清空主页面旧内容
    print("[~] Clearing old dashboard blocks...")
    children = notion.blocks.children.list(NOTION_SUMMARY_PAGE_ID).get("results", [])
    for child in children:
        notion.blocks.delete(child["id"])

    # 写入新内容
    print("[*] Uploading new dashboard blocks...")
    notion.blocks.children.append(NOTION_SUMMARY_PAGE_ID, children=all_blocks)

    print("[✅] Dashboard & Database updated successfully!")


if __name__ == "__main__":
    upload_to_notion()
