#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload crypto CSI analysis results to Notion database
------------------------------------------------------
Pushes CSV and PNG files in docs/ to a Notion database.
Requires NOTION_TOKEN and NOTION_DATABASE_ID as environment variables.
"""

import os, re, time, glob
from notion_client import Client

notion = Client(auth=os.environ["NOTION_TOKEN"])
db_id = os.environ["NOTION_DATABASE_ID"]

docs_dir = "docs"
files = sorted(glob.glob(os.path.join(docs_dir, "*.png")))

if not files:
    print("[!] No PNG files found in docs/. Nothing to upload.")
    exit(0)

for png_path in files:
    base = os.path.basename(png_path)
    token_match = re.search(r"Binance_(.*?)_", base)
    timeframe_match = re.search(r"_(\d+h)_", base)

    token = token_match.group(1) if token_match else "Unknown"
    timeframe = timeframe_match.group(1) if timeframe_match else "Unknown"

    csv_path = png_path.replace("_chip_timeline_pro.png", "_chip_strength.csv")

    # Build GitHub raw URLs
    csv_url = f"https://raw.githubusercontent.com/CMUJIN/crypto_csi_toolkit/main/{csv_path}"
    png_url = f"https://raw.githubusercontent.com/CMUJIN/crypto_csi_toolkit/main/{png_path}"

    # Create / update record in Notion
    notion.pages.create(
        parent={"database_id": db_id},
        properties={
            "Token": {"title": [{"text": {"content": token}}]},
            "Timeframe": {"rich_text": [{"text": {"content": timeframe}}]},
            "Updated": {"date": {"start": time.strftime("%Y-%m-%dT%H:%M:%S")}},
            "Strength CSV": {"url": csv_url},
        },
        children=[
            {
                "object": "block",
                "type": "image",
                "image": {"external": {"url": png_url}},
            }
        ],
    )

    print(f"[OK] Uploaded {token} ({timeframe}) to Notion.")
