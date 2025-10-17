#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_to_notion.py (Secrets-safe Stable Version)
-------------------------------------------------
✅ 不再硬编码 Notion ID
✅ 从 GitHub Secrets 读取 NOTION_TOKEN / NOTION_DATABASE_ID / NOTION_SUMMARY_PAGE_ID
✅ 仅归档旧记录，不删除数据库
✅ 更新汇总页面图表与表格（Top 20% 筹码区）
"""

import os, sys, pandas as pd
from notion_client import Client

# ======== CONFIG ========
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID")
NOTION_SUMMARY_PAGE_ID = os.environ.get("NOTION_SUMMARY_PAGE_ID")

if not NOTION_TOKEN or not NOTION_DATABASE_ID or not NOTION_SUMMARY_PAGE_ID:
    sys.exit("[X] Missing required Notion secrets. Please check GitHub Secrets configuration.")

notion = Client(auth=NOTION_TOKEN)

# ======== HELPERS ========
def fmt_price(v: float) -> str:
    """格式化价格：大于100取整，否则两位小数"""
    try:
        v = float(v)
        return f"{int(round(v))}" if v >= 100 else f"{v:.2f}"
    except:
        return str(v)

def clear_summary_blocks():
    """清空汇总页旧内容"""
    print("[~] Clearing old summary blocks...")
    try:
        blocks = notion.blocks.children.list(NOTION_SUMMARY_PAGE_ID).get("results", [])
        for blk in blocks:
            notion.blocks.delete(blk["id"])
        print(f"[OK] Summary cleared: {len(blocks)} blocks removed")
    except Exception as e:
        print(f"[!] Failed to clear summary: {e}")

def find_page_by_token(symbol: str):
    """按 Token 查找数据库记录"""
    try:
        res = notion.databases.query(
            NOTION_DATABASE_ID,
            filter={"property": "Token", "title": {"equals": symbol}}
        )
        if res.get("results"):
            return res["results"][0]["id"]
    except Exception as e:
        print(f"[!] find_page_by_token({symbol}) error: {e}")
    return None

def archive_old_records():
    """归档旧数据库记录"""
    print("[~] Archiving old database records...")
    try:
        res = notion.databases.query(NOTION_DATABASE_ID)
        for page in res.get("results", []):
            notion.pages.update(page["id"], archived=True)
        print("[OK] Old records archived")
    except Exception as e:
        print(f"[!] Failed to archive old records: {e}")

def create_or_update_record(symbol, chart_url, csv_url):
    """更新或创建数据库记录"""
    pid = find_page_by_token(symbol)
    props = {
        "Token": {"title": [{"text": {"content": symbol}}]},
        "Chart": {"url": chart_url},
        "CSV": {"url": csv_url},
    }
    try:
        if pid:
            notion.pages.update(pid, properties=props)
            print(f"[~] Updated record: {symbol}")
        else:
            notion.pages.create(parent={"database_id": NOTION_DATABASE_ID}, properties=props)
            print(f"[+] Created new record: {symbol}")
    except Exception as e:
        print(f"[X] Failed to update/create record for {symbol}: {e}")

def build_table_block(df: pd.DataFrame):
    """生成三列表格块：low, high, strength"""
    if df.empty:
        return []
    df = df[["low", "high", "strength"]].head(30)
    header = ["low", "high", "strength"]

    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in header:
            val = fmt_price(row[col]) if col in ("low", "high") else f"{row[col]:.3f}"
            cells.append({"type": "text", "text": {"content": val}})
        rows.append({"type": "table_row", "cells": [[c] for c in cells]})

    return [{
        "object": "block",
        "type": "table",
        "table": {
            "table_width": len(header),
            "has_column_header": True,
            "has_row_header": False,
            "children": rows
        }
    }]

def update_summary(chip_data):
    """重建汇总页"""
    clear_summary_blocks()
    children = []

    for item in chip_data:
        symbol, chart_url, chip_csv = item["symbol"], item["chart_url"], item["csv_path"]

        children.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": symbol}}]}
        })
        children.append({
            "object": "block",
            "type": "image",
            "image": {"type": "external", "external": {"url": chart_url}}
        })

        try:
            df = pd.read_csv(chip_csv)
            top_df = df.sort_values("strength", ascending=False)
            threshold = top_df["strength"].quantile(0.8)
            filtered = top_df[top_df["strength"] >= threshold].head(30)
            children.extend(build_table_block(filtered))
        except Exception as e:
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": f"[X] Failed to load table: {e}"}}]}
            })

    notion.blocks.children.append(NOTION_SUMMARY_PAGE_ID, children={"children": children})
    print(f"[OK] Summary updated: {len(chip_data)} items")

# ======== MAIN ========
def upload_to_notion():
    print("[*] Notion sync start...")
    data_items = []

    for file in os.listdir("docs"):
        if file.endswith("_chip_timeline_pro.png"):
            symbol = file.split("_")[1]
            chart_url = f"https://cmujin.github.io/crypto_csi_toolkit/{file}"
            csv_path = os.path.join("docs", file.replace("_chip_timeline_pro.png", "_chip_strength.csv"))
            csv_url = f"https://cmujin.github.io/crypto_csi_toolkit/{os.path.basename(csv_path)}"
            if os.path.exists(csv_path):
                data_items.append({"symbol": symbol, "chart_url": chart_url, "csv_path": csv_path, "csv_url": csv_url})

    if not data_items:
        print("[!] No data found, abort.")
        return

    print(f"[OK] Found {len(data_items)} chip_strength CSVs.")
    archive_old_records()

    for item in data_items:
        create_or_update_record(item["symbol"], item["chart_url"], item["csv_url"])

    update_summary(data_items)
    print("[OK] Database & summary page updated successfully!")

if __name__ == "__main__":
    upload_to_notion()
