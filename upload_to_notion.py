import os
import csv
from notion_client import Client
from datetime import datetime, timezone
from pathlib import Path

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

notion = Client(auth=NOTION_TOKEN)
repo_url_base = "https://cmujin.github.io/crypto_csi_toolkit"


def notion_url(path: str) -> str:
    ts = int(datetime.now().timestamp())
    return f"{repo_url_base}/{path}?ver={ts}"


def find_page_id_by_symbol(symbol: str):
    query = notion.databases.query(
        **{
            "database_id": DATABASE_ID,
            "filter": {"property": "Name", "title": {"equals": symbol}},
        }
    )
    results = query.get("results", [])
    return results[0]["id"] if results else None


def clear_old_blocks(page_id):
    try:
        children = notion.blocks.children.list(page_id)
        for child in children.get("results", []):
            notion.blocks.delete(child["id"])
        print(f"[~] Cleared old blocks for {page_id}")
    except Exception as e:
        print(f"[!] Could not clear old blocks ({e})")


def parse_csv_to_table(csv_path: Path, max_rows=10):
    """将 CSV 转为 Notion 表格格式"""
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        headers = reader[0]
        rows = reader[1:max_rows + 1]
        if len(reader) > max_rows + 1:
            rows.append(["..."] * len(headers))
    return headers, rows


def add_visual_report(page_id, symbol, png_url, csv_path, csv_url):
    try:
        headers, rows = parse_csv_to_table(csv_path)

        # 构造表格行
        table_rows = []
        # header
        table_rows.append({
            "object": "block",
            "type": "table_row",
            "table_row": {
                "cells": [[{"type": "text", "text": {"content": h}}] for h in headers]
            },
        })
        # data rows
        for row in rows:
            table_rows.append({
                "object": "block",
                "type": "table_row",
                "table_row": {
                    "cells": [[{"type": "text", "text": {"content": str(c)}}] for c in row]
                },
            })

        # 主体块
        blocks = [
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
                    "children": table_rows,  # ✅ 一次性附带表格行
                },
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {"type": "text", "text": {"content": "📎 查看完整CSV文件：", "annotations": {"bold": True}}},
                        {"type": "text", "text": {"content": csv_url, "link": {"url": csv_url}}},
                    ]
                },
            },
        ]

        # 一次性写入所有块
        notion.blocks.children.append(page_id, children=blocks)
        print(f"[OK] Added table report for {symbol}")

    except Exception as e:
        print(f"[X] Failed to add report for {symbol}: {e}")


def upload_to_notion():
    docs_dir = Path("docs")
    csv_files = list(docs_dir.glob("Binance_*_to_latest.csv"))

    for csv_file in csv_files:
        symbol = csv_file.name.split("_")[1]
        png_file = csv_file.with_name(csv_file.stem + "_chip_timeline_pro.png")
        csv_url = notion_url(csv_file.name)
        png_url = notion_url(png_file.name)

        print(f"[*] Syncing {symbol} to Notion...")

        page_id = find_page_id_by_symbol(symbol)
        if not page_id:
            print(f"[+] Creating new page for {symbol}")
            new_page = notion.pages.create(
                **{
                    "parent": {"database_id": DATABASE_ID},
                    "properties": {
                        "Name": {"title": [{"text": {"content": symbol}}]},
                        "Timeframe": {"rich_text": [{"text": {"content": "1h"}}]},
                        "Updated": {"date": {"start": datetime.now(timezone.utc).isoformat()}},
                        "CSV": {"url": csv_url},
                        "Chart": {"url": png_url},
                    },
                }
            )
            page_id = new_page["id"]
        else:
            print(f"[~] Page exists for {symbol}, updating...")
            notion.pages.update(
                **{
                    "page_id": page_id,
                    "properties": {
                        "Updated": {"date": {"start": datetime.now(timezone.utc).isoformat()}},
                        "CSV": {"url": csv_url},
                        "Chart": {"url": png_url},
                    },
                }
            )

        clear_old_blocks(page_id)
        add_visual_report(page_id, symbol, png_url, csv_file, csv_url)
        print(f"[OK] Updated {symbol} page complete ✅")


if __name__ == "__main__":
    upload_to_notion()
