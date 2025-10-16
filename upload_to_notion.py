import os
from datetime import datetime
from notion_client import Client

# === 环境变量 ===
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID")

notion = Client(auth=NOTION_TOKEN)

REPO = "CMUJIN/crypto_csi_toolkit"
BRANCH = "main"
DOCS_DIR = "docs"

# === 生成可访问链接 ===
def github_raw_url(filename):
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{DOCS_DIR}/{filename}?t={ts}"

def github_file_url(filename):
    return f"https://github.com/{REPO}/blob/{BRANCH}/{DOCS_DIR}/{filename}"

# === 检查 Notion 是否已有页面 ===
def find_page_id_by_symbol(symbol):
    query = notion.databases.query(
        **{
            "database_id": NOTION_DATABASE_ID,
            "filter": {
                "property": "Name",
                "title": {"equals": symbol}
            }
        }
    )
    results = query.get("results", [])
    if results:
        return results[0]["id"]
    return None

# === 上传或更新页面 ===
def upload_to_notion():
    if not NOTION_TOKEN or not NOTION_DATABASE_ID:
        print("[X] Missing NOTION_TOKEN or NOTION_DATABASE_ID.")
        return

    files = os.listdir(DOCS_DIR)
    csvs = [f for f in files if f.endswith(".csv")]
    pngs = [f for f in files if f.endswith(".png")]

    for csv in csvs:
        symbol = csv.split("_")[1]
        png_match = next((p for p in pngs if symbol in p), None)

        csv_url = github_file_url(csv)
        png_url = github_raw_url(png_match) if png_match else None

        print(f"[*] Syncing {symbol} to Notion...")

        children_blocks = []
        if png_url:
            children_blocks.append({
                "object": "block",
                "type": "image",
                "image": {"type": "external", "external": {"url": png_url}}
            })
        children_blocks.append({
            "object": "block",
            "type": "embed",
            "embed": {"url": csv_url}
        })

        # === 检查是否已存在 ===
        page_id = find_page_id_by_symbol(symbol)

        if page_id:
            print(f"[~] Page exists for {symbol}, updating...")
            # 先清空旧 blocks
            children = notion.blocks.children.list(page_id).get("results", [])
            for block in children:
                try:
                    notion.blocks.delete(block["id"])
                except Exception as e:
                    print(f"  [!] Warning: failed to delete block: {e}")

            # 再更新页面内容
            notion.pages.update(
                page_id=page_id,
                properties={
                    "CSV": {"url": csv_url},
                }
            )
            notion.blocks.children.append(page_id, children=children_blocks)
            print(f"[OK] Updated Notion page for {symbol}")

        else:
            print(f"[+] Creating new page for {symbol}")
            notion.pages.create(
                parent={"database_id": NOTION_DATABASE_ID},
                properties={
                    "Name": {"title": [{"text": {"content": symbol}}]},
                    "CSV": {"url": csv_url},
                },
                children=children_blocks
            )
            print(f"[OK] Created new Notion page for {symbol}")

if __name__ == "__main__":
    upload_to_notion()
