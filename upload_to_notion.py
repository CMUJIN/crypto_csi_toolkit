import os
from notion_client import Client
from datetime import datetime, timezone
from pathlib import Path
import time

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
            "filter": {"property": "Name", "title": {"equals": symbol}},
        }
    )
    results = query.get("results", [])
    if results:
        return results[0]["id"]
    return None


def clear_old_blocks(page_id):
    """Clear old content blocks under page (if any)"""
    try:
        children = notion.blocks.children.list(page_id)
        for child in children.get("results", []):
            notion.blocks.delete(child["id"])
        print(f"[~] Cleared old blocks for page {page_id}")
    except Exception as e:
        print(f"[!] Warning: Could not clear old blocks ({e})")


def add_visual_blocks(page_id, png_url, csv_url):
    """Insert grouped visual content into page"""
    try:
        # Step 1. Create a parent "📊 Content" toggle block
        content_block = notion.blocks.children.append(
            page_id,
            children=[
                {
                    "object": "block",
                    "type": "toggle",
                    "toggle": {
                        "rich_text": [{"type": "text", "text": {"content": "📊 Content"}}],
                        "children": [],
                    },
                }
            ],
        )

        parent_id = content_block["results"][0]["id"]
        print(f"[OK] Created '📊 Content' block with ID: {parent_id}")

        # Step 2. Add sub-blocks under Content
        sub_blocks = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "Latest Chart"}}]
                },
            },
            {
                "object": "block",
                "type": "image",
                "image": {"type": "external", "external": {"url": png_url}},
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "CSV Data"}}]
                },
            },
            {
                "object": "block",
                "type": "embed",
                "embed": {"url": csv_url},
            },
        ]

        time.sleep(1)
        notion.blocks.children.append(parent_id, children=sub_blocks)
        print(f"[OK] Added PNG + CSV blocks under '📊 Content' for {page_id}")

    except Exception as e:
        print(f"[X] Failed to add visual blocks: {e}")


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
                        "Timeframe": {
                            "rich_text": [{"text": {"content": "1h"}}]
                        },
                        "Updated": {
                            "date": {"start": datetime.now(timezone.utc).isoformat()}
                        },
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
                        "Updated": {
                            "date": {"start": datetime.now(timezone.utc).isoformat()}
                        },
                        "CSV": {"url": csv_url},
                        "Chart": {"url": png_url},
                    },
                }
            )

        # 🧹 Clear old blocks + Insert new grouped content
        clear_old_blocks(page_id)
        add_visual_blocks(page_id, png_url, csv_url)

        print(f"[OK] Updated {symbol} | PNG={png_url} | CSV={csv_url}")


if __name__ == "__main__":
    upload_to_notion()
