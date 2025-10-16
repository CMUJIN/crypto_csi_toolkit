import os
import pandas as pd
from notion_client import Client
from datetime import datetime, timezone

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
NOTION_SUMMARY_PAGE_ID = os.getenv("NOTION_SUMMARY_PAGE_ID", "").strip()

notion = Client(auth=NOTION_TOKEN)

def upload_to_notion():
    print("[*] Starting Notion sync...")

    # 1️⃣ 查找 CSV 文件
    csv_files = [f for f in os.listdir("docs") if f.endswith(".csv")]
    print(f"[*] Found {len(csv_files)} CSV files to sync.")

    for csv_file in csv_files:
        symbol = csv_file.split("_")[1] if "_" in csv_file else csv_file
        csv_path = f"docs/{csv_file}"
        png_path = csv_path.replace(".csv", "_chip_timeline_pro.png")

        # 生成 GitHub Pages URL
        base_url = "https://cmujin.github.io/crypto_csi_toolkit/"
        timestamp = int(datetime.now().timestamp())
        csv_url = f"{base_url}{csv_file}?ver={timestamp}"
        png_url = f"{base_url}{os.path.basename(png_path)}?ver={timestamp}"

        # 读取 CSV 数据
        df = pd.read_csv(csv_path)

        # 🧮 动态选取显示区间（前20%强度，最多30行）
        if "strength" in df.columns:
            threshold = df["strength"].quantile(0.8)
            highlight_rows = df[df["strength"] >= threshold]
            rows = highlight_rows if len(highlight_rows) <= 30 else highlight_rows.head(30)
        else:
            rows = df.head(30)

        # 格式化时间
        now_str = datetime.now(timezone.utc).astimezone().strftime("%B %d, %Y %I:%M %p")

        # 2️⃣ 更新数据库条目
        pages = notion.databases.query(
            **{
                "database_id": NOTION_DATABASE_ID,
                "filter": {"property": "Name", "title": {"equals": symbol}},
            }
        ).get("results", [])

        if pages:
            page_id = pages[0]["id"]
            print(f"[~] Updated {symbol} in database ({now_str})")
            notion.pages.update(
                page_id=page_id,
                properties={
                    "Last Updated": {"date": {"start": datetime.now(timezone.utc).isoformat()}},
                    "Chart": {"url": png_url},
                    "CSV": {"url": csv_url},
                },
            )
        else:
            print(f"[+] Creating new entry for {symbol}")
            notion.pages.create(
                parent={"database_id": NOTION_DATABASE_ID},
                properties={
                    "Name": {"title": [{"text": {"content": symbol}}]},
                    "Last Updated": {"date": {"start": datetime.now(timezone.utc).isoformat()}},
                    "Chart": {"url": png_url},
                    "CSV": {"url": csv_url},
                },
            )

    # 3️⃣ 更新主汇总页面
    if NOTION_SUMMARY_PAGE_ID:
        print("[~] Updating main summary dashboard...")

        # 清空旧内容
        notion.blocks.children.append(
            NOTION_SUMMARY_PAGE_ID,
            {
                "children": []
            }
        )

        # 构建表格块
        table_children = []
        for csv_file in csv_files:
            symbol = csv_file.split("_")[1]
            csv_path = f"docs/{csv_file}"
            png_path = csv_path.replace(".csv", "_chip_timeline_pro.png")
            png_url = f"{base_url}{os.path.basename(png_path)}?ver={timestamp}"
            df = pd.read_csv(csv_path)

            # 同样的筛选逻辑
            if "strength" in df.columns:
                threshold = df["strength"].quantile(0.8)
                highlight_rows = df[df["strength"] >= threshold]
                rows = highlight_rows if len(highlight_rows) <= 30 else highlight_rows.head(30)
            else:
                rows = df.head(30)

            # 添加图片
            table_children.append({
                "object": "block",
                "type": "image",
                "image": {
                    "type": "external",
                    "external": {"url": png_url}
                }
            })

            # 添加表格
            table_rows = []
            for _, row in rows.iterrows():
                cols = [str(v)[:12] for v in row.values[:6]]
                table_rows.append({
                    "object": "block",
                    "type": "table_row",
                    "table_row": {
                        "cells": [[{"text": {"content": c}}] for c in cols]
                    }
                })

            if table_rows:
                table_children.append({
                    "object": "block",
                    "type": "table",
                    "table": {
                        "has_column_header": False,
                        "has_row_header": False,
                        "table_width": len(rows.columns[:6]),
                        "children": table_rows
                    }
                })

        # 上传汇总内容
        notion.blocks.children.append(
            NOTION_SUMMARY_PAGE_ID,
            {"children": table_children}
        )

        print("[OK] Dashboard updated successfully.")

    print("[✓] All sync operations completed.")


if __name__ == "__main__":
    upload_to_notion()
