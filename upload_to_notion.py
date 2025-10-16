# -*- coding: utf-8 -*-
"""
upload_to_notion.py (final stable version)
------------------------------------------
Features:
1. Upload only *_chip_strength.csv results.
2. Show Top 20% zones in Notion summary (≤30 rows).
3. Display only [low, high, strength] columns.
4. Format prices intelligently.
"""

import os
import pandas as pd
from notion_client import Client
from datetime import datetime, timezone

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
NOTION_SUMMARY_PAGE_ID = os.getenv("NOTION_SUMMARY_PAGE_ID")

notion = Client(auth=NOTION_TOKEN)

# -----------------------------
#  Helper functions
# -----------------------------
def fmt_price(v):
    """Format low/high values based on magnitude"""
    try:
        v = float(v)
        return f"{v:.0f}" if v >= 100 else f"{v:.2f}"
    except Exception:
        return str(v)

def read_top_chip_strength(csv_path, top_pct=0.2, max_rows=30):
    """Read chip_strength.csv and select top N%"""
    df = pd.read_csv(csv_path)
    df = df[[c for c in df.columns if c.lower() in ["low", "high", "strength"]]]
    df = df.sort_values("strength", ascending=False)
    top_n = int(len(df) * top_pct)
    top_n = min(max_rows, max(1, top_n))
    return df.head(top_n)

def clear_summary_page_blocks(page_id):
    """Delete all blocks under summary page before re-adding"""
    children = notion.blocks.children.list(page_id).get("results", [])
    for c in children:
        try:
            notion.blocks.delete(c["id"])
        except Exception:
            pass

def build_symbol_section_blocks(symbol, csv_path, chart_url):
    """Create heading + image + 3-column Top20% chip_strength table"""
    df = read_top_chip_strength(csv_path, top_pct=0.2, max_rows=30)
    df["low"] = df["low"].apply(fmt_price)
    df["high"] = df["high"].apply(fmt_price)

    headers = ["low", "high", "strength"]
    header_row = {
        "object": "block",
        "type": "table_row",
        "table_row": {
            "cells": [[{"type": "text", "text": {"content": h}}] for h in headers]
        },
    }

    data_rows = []
    for _, row in df.iterrows():
        cells = [[{"type": "text", "text": {"content": str(row[col])}}] for col in headers]
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
        "heading_2": {
            "rich_text": [
                {"type": "text", "text": {"content": f"{symbol} Chip Strength (Top 20%)"}}
            ]
        },
    }

    return [heading_block, image_block, table_block]


# -----------------------------
#  Main upload logic
# -----------------------------
def upload_to_notion():
    print("[*] Notion sync start...")

    items = []
    for fn in sorted(os.listdir("docs")):
        if not fn.endswith("_chip_strength.csv"):
            continue

        csv_path = os.path.join("docs", fn)
        symbol = fn.split("_")[1].upper()

        chart_name = fn.replace("_chip_strength.csv", "_chip_timeline_pro.png")
        chart_url = f"https://cmujin.github.io/crypto_csi_toolkit/{chart_name}"

        items.append({
            "symbol": symbol,
            "csv_path": csv_path,
            "chart_url": chart_url,
        })

    print(f"[OK] Found {len(items)} chip_strength CSVs.")

    # clear previous blocks (do NOT delete database)
    print("[~] Clearing old blocks from summary page...")
    clear_summary_page_blocks(NOTION_SUMMARY_PAGE_ID)

    # rebuild blocks for each symbol
    new_blocks = []
    for it in items:
        new_blocks += build_symbol_section_blocks(it["symbol"], it["csv_path"], it["chart_url"])

    # upload blocks to summary page
    notion.blocks.children.append(block_id=NOTION_SUMMARY_PAGE_ID, children=new_blocks)
    print("[OK] Summary page updated successfully!")

    # update database records (overwrite previous entries)
    for it in items:
        update_or_create_database_record(it["symbol"], it["chart_url"])
    print("[OK] Database updated successfully.")


def update_or_create_database_record(symbol, chart_url):
    """Overwrite database record (do not delete database)"""
    try:
        pages = notion.databases.query(
            NOTION_DATABASE_ID,
            filter={"property": "Token", "title": {"equals": symbol}},
        )
        if pages["results"]:
            page_id = pages["results"][0]["id"]
            notion.pages.update(
                page_id=page_id,
                properties={
                    "Updated": {"date": {"start": datetime.now(timezone.utc).isoformat()}},
                    "Chart": {"url": chart_url},
                },
            )
        else:
            notion.pages.create(
                parent={"database_id": NOTION_DATABASE_ID},
                properties={
                    "Token": {"title": [{"text": {"content": symbol}}]},
                    "Chart": {"url": chart_url},
                    "Updated": {"date": {"start": datetime.now(timezone.utc).isoformat()}},
                },
            )
    except Exception as e:
        print(f"[X] Failed to update DB record for {symbol}: {e}")


if __name__ == "__main__":
    upload_to_notion()
