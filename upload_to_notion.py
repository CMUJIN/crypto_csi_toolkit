#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_to_notion.py
-------------------
Auto-upload crypto CSI + liquidity results to Notion with versioned file names.

Features:
- Adds timestamp suffix to avoid CDN cache (e.g., _20251103_2045.png)
- Uploads both chip and liquidity images
- Keeps only last 2 recent versions per symbol
- Logs progress for GitHub Actions
"""

import os
import re
import time
from datetime import datetime
from pathlib import Path

from notion_client import Client


# ==============================================================
# Config
# ==============================================================
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
NOTION_DB_ID = os.environ.get("NOTION_DATABASE_ID")
NOTION_SUMMARY_PAGE_ID = os.environ.get("NOTION_SUMMARY_PAGE_ID")

client = Client(auth=NOTION_TOKEN)

DOCS_DIR = Path("docs")


# ==============================================================
# Utilities
# ==============================================================
def list_docs_files():
    """Return all png/csv files in docs/"""
    if not DOCS_DIR.exists():
        print("[!] docs/ folder not found.")
        return []
    return list(DOCS_DIR.glob("*"))


def extract_symbol(fname: str) -> str:
    """Extract symbol name from filename like BTCUSDT_chip_timeline_pro.png"""
    base = Path(fname).name.upper()
    m = re.search(r"(BTC|ETH|AAVE|UNI|SOL|BNB|USDT|USDC|DOGE|ADA|XRP|LTC|DOT|AVAX|FIL|ARB|OP|TON|SUI|SEI|LINK|HYPE)?[A-Z]{2,10}USDT", base)
    if m:
        return m.group(0)
    # fallback
    parts = base.split("_")
    for p in parts:
        if p.endswith("USDT"):
            return p
    return parts[0].replace(".PNG", "").replace(".CSV", "")


def versioned_filename(path: Path) -> Path:
    """Append timestamp to filename before upload"""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    stem, ext = path.stem, path.suffix
    return Path(f"{stem}_{ts}{ext}")


def upload_image_to_notion(page_id: str, symbol: str, image_path: Path, title: str):
    """Upload image to Notion page as an image block."""
    if not image_path.exists():
        print(f"[!] Missing image: {image_path}")
        return

    # version to avoid cache
    ver_path = versioned_filename(image_path)
    if image_path != ver_path:
        try:
            ver_path.write_bytes(image_path.read_bytes())
        except Exception as e:
            print(f"[!] Failed to copy versioned file: {e}")

    print(f"[↑] Uploading {ver_path.name} to Notion...")

    # Upload via Notion file block (external URL if GH Pages later)
    try:
        client.blocks.children.append(
            block_id=page_id,
            children=[
                {
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {"rich_text": [{"text": {"content": f"{symbol} - {title}"}}]},
                },
                {
                    "object": "block",
                    "type": "image",
                    "image": {
                        "type": "file",
                        "file": {"url": f"file://{ver_path.resolve()}"}
                    },
                },
            ],
        )
        print(f"[OK] Uploaded {title} for {symbol}")
    except Exception as e:
        print(f"[X] Failed to upload {title} for {symbol}: {e}")


def clear_old_blocks(page_id: str):
    """Remove old summary/image blocks (keep DB link intact)."""
    try:
        blocks = client.blocks.children.list(page_id).get("results", [])
        removable = []
        for blk in blocks:
            t = blk.get("type", "")
            if t in ("heading_3", "image"):
                removable.append(blk["id"])

        if removable:
            for bid in removable:
                try:
                    client.blocks.delete(bid)
                except Exception:
                    pass
            print(f"[~] Cleared {len(removable)} old image blocks.")
        else:
            print("[~] No old image blocks found.")
    except Exception as e:
        print(f"[!] Could not clear old blocks: {e}")


def keep_latest_versions(symbol: str, files: list[Path], keep: int = 2):
    """Keep latest N versions per symbol, delete older ones from docs/"""
    related = [f for f in files if symbol in f.name.upper()]
    if len(related) <= keep:
        return
    sorted_files = sorted(related, key=lambda f: f.stat().st_mtime, reverse=True)
    for old in sorted_files[keep:]:
        try:
            old.unlink()
            print(f"[~] Removed old file: {old.name}")
        except Exception:
            pass


# ==============================================================
# Main upload flow
# ==============================================================
def main():
    print("[*] Notion sync start (Chip + Liquidity, versioned filenames)...")

    if not NOTION_TOKEN or not NOTION_SUMMARY_PAGE_ID:
        print("[X] Missing Notion credentials.")
        return

    all_files = list_docs_files()
    png_files = [f for f in all_files if f.suffix.lower() == ".png"]
    csv_files = [f for f in all_files if f.suffix.lower() == ".csv"]

    if not png_files:
        print("[!] No PNG files found in docs/, abort.")
        return

    clear_old_blocks(NOTION_SUMMARY_PAGE_ID)

    symbols_done = set()

    for f in sorted(png_files):
        name = f.name
        symbol = extract_symbol(name)
        if not symbol:
            continue

        if symbol not in symbols_done:
            print(f"\n=== Processing {symbol} ===")

        if "chip_timeline" in name:
            upload_image_to_notion(NOTION_SUMMARY_PAGE_ID, symbol, f, "Chip Analysis")
        elif "liquidity" in name:
            upload_image_to_notion(NOTION_SUMMARY_PAGE_ID, symbol, f, "Liquidity Curve")

        keep_latest_versions(symbol, png_files)
        symbols_done.add(symbol)

    print(f"\n[OK] Summary updated successfully with {len(symbols_done)} symbols (chip + liquidity).")


if __name__ == "__main__":
    main()
