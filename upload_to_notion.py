import os
from notion_client import Client
from datetime import datetime, timezone
from pathlib import Path

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

notion = Client(auth=NOTION_TOKEN)
repo_url_base = "https://cmujin.github.io/crypto_csi_toolkit"

def notion_url(path: str) -> str:
    """Generate GitHub Pages URL with timestamp to avoid cache"""
    ts = int(datetime.now().timestamp())
    return f"{repo_url_base}/{path}?ver={ts}"

def find_page_id_by_symbol(symbol: str):
    """Find Notion page id by Name"""
    query = notion.databases.query(
        **{
            "database_id": DATABASE_ID,
            "filter": {
                "property": "Name",
                "title": {"equals": symbol}
            },
        }
    )
    results = query.get("results", [])
    if results:
        return results[0]["id"]
    return None

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
            notion.pages.create(
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
        print(f"[OK] Updated {symbol} | CSV={csv_url} | Chart={png_url}")

if __name__ == "__main__":
    upload_to_notion()
