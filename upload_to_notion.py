# -*- coding: utf-8 -*-
"""
upload_to_notion.py  (v2.4)
将 Crypto CSI 分析结果（CSV + PNG）同步上传到 Notion 数据库。
URL 采用 GitHub Pages 域名结构：
  https://<username>.github.io/<repo>/<subdir>/<file>?ver=<timestamp>
"""

import os
import datetime
from notion_client import Client

# === 环境变量 ===
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

notion = Client(auth=NOTION_TOKEN)

# === GitHub Pages 配置 ===
USERNAME = "CMUJIN"
REPO = "crypto_csi_toolkit"
SUBDIR = "data"  # GitHub Pages 对应 docs/data/ 目录
BASE_URL = f"https://{USERNAME}.github.io/{REPO}/{SUBDIR}"

def find_page_id_by_symbol(symbol):
    """查找数据库中对应 symbol 的页面 ID"""
    query = notion.databases.query(
        database_id=NOTION_DATABASE_ID,
        filter={"property": "Name", "title": {"equals": symbol}}
    )
    results = query.get("results", [])
    return results[0]["id"] if results else None


def upload_to_notion():
    symbols = ["ETHUSDT", "BTCUSDT"]
    timeframe = "1h"

    for symbol in symbols:
        print(f"[*] Syncing {symbol} to Notion...")
        page_id = find_page_id_by_symbol(symbol)

        # === 构建文件名 ===
        csv_filename = f"Binance_{symbol}_{timeframe}_2025-08-01_to_latest_chip_strength.csv"
        png_filename = f"Binance_{symbol}_{timeframe}_2025-08-01_to_latest_chip_timeline_pro.png"

        # === GitHub Pages URL + 时间戳 ===
        ver = int(datetime.datetime.now().timestamp())
        csv_url = f"{BASE_URL}/{csv_filename}?ver={ver}"
        chart_url = f"{BASE_URL}/{png_filename}?ver={ver}"

        # === 上传或更新 Notion 页面 ===
        props = {
            "Timeframe": {"rich_text": [{"text": {"content": timeframe}}]},
            "Updated": {"date": {"start": datetime.datetime.now().isoformat()}},
            "CSV": {"url": csv_url},
            "Chart": {"url": chart_url},
        }

        if not page_id:
            print(f"[+] Creating new page for {symbol}...")
            notion.pages.create(
                parent={"database_id": NOTION_DATABASE_ID},
                properties={"Name": {"title": [{"text": {"content": symbol}}]}, **props},
            )
        else:
            print(f"[~] Page exists for {symbol}, updating...")
            notion.pages.update(page_id=page_id, properties=props)
            print(f"[✓] Updated Notion entry for {symbol}")

    print("\n✅ All symbols synced to Notion successfully!")


if __name__ == "__main__":
    upload_to_notion()
