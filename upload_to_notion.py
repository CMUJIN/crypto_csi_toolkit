# -*- coding: utf-8 -*-
"""
upload_to_notion_v2.6.py
--------------------------------------
同步最新分析结果至 Notion 数据库
功能：
- 自动上传 CSV 和 PNG 文件的公开链接（GitHub Pages）
- 自动更新时间戳防缓存
- 自动更新已有页或创建新页
"""

import os
import time
from notion_client import Client
from datetime import datetime, timezone

# === 配置区 ===
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
BASE_URL = "https://cmujin.github.io/crypto_csi_toolkit"  # GitHub Pages 根目录

notion = Client(auth=NOTION_TOKEN)


# === 查找 Notion 页面 ===
def find_page_id_by_name(name):
    query = notion.databases.query(
        **{
            "database_id": DATABASE_ID,
            "filter": {"property": "Name", "title": {"equals": name}},
        }
    )
    results = query.get("results", [])
    return results[0]["id"] if results else None


# === 创建或更新页面 ===
def upsert_page(symbol, timeframe, csv_url, chart_url):
    now_str = datetime.now(timezone.utc).isoformat()
    page_id = find_page_id_by_name(symbol)

    if page_id:
        print(f"[~] Updating existing Notion page for {symbol} ...")
        notion.pages.update(
            page_id=page_id,
            properties={
                "Timeframe": {"rich_text": [{"text": {"content": timeframe}}]},
                "Updated": {"date": {"start": now_str}},
                "CSV": {"url": csv_url},
                "Chart": {"url": chart_url},
            },
        )
    else:
        print(f"[+] Creating new Notion page for {symbol} ...")
        notion.pages.create(
            parent={"database_id": DATABASE_ID},
            properties={
                "Name": {"title": [{"text": {"content": symbol}}]},
                "Timeframe": {"rich_text": [{"text": {"content": timeframe}}]},
                "Updated": {"date": {"start": now_str}},
                "CSV": {"url": csv_url},
                "Chart": {"url": chart_url},
            },
        )


# === 主执行函数 ===
def upload_to_notion():
    print("[*] Starting Notion sync...")
    timestamp = int(time.time())

    for file in os.listdir("docs"):
        if not file.endswith(".csv"):
            continue
        try:
            symbol, freq, *_ = file.replace("Binance_", "").split("_")
            csv_file = file
            png_file = file.replace("_chip_strength.csv", "_chip_timeline_pro.png")

            csv_path = f"{BASE_URL}/{csv_file}?ver={timestamp}"
            png_path = f"{BASE_URL}/{png_file}?ver={timestamp}"

            print(f"[*] Syncing {symbol} ({freq}) to Notion...")
            upsert_page(symbol, freq, csv_path, png_path)
            print(f"[OK] {symbol} synced successfully.\n")
        except Exception as e:
            print(f"[X] Failed to sync {file}: {e}")


if __name__ == "__main__":
    upload_to_notion()
