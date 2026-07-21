"""
Fetch 10 Wikipedia articles on Alzheimer's-related topics via the Wikipedia
Action API (plain-text extracts). Saves JSON + plain-text files into
'Selected articles/'.
"""
import json
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path(__file__).resolve().parent.parent / "Selected articles"
OUT_DIR.mkdir(exist_ok=True)

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "HybridRAG-dataset-builder/1.0 (research use; contact: prabhat0405005@gmail.com)"}

# Chosen to overlap conceptually with the PubMed subtopics (shared entities
# like amyloid-beta, tau, APOE make the knowledge graph well-connected).
TITLES = [
    "Alzheimer's disease",
    "Amyloid beta",
    "Tau protein",
    "Apolipoprotein E",
    "Dementia",
    "Mild cognitive impairment",
    "Cholinesterase inhibitor",
    "Memantine",
    "Neurofibrillary tangle",
    "Lecanemab",
]


def slugify(title):
    return title.lower().replace("'", "").replace(" ", "_")


def fetch_article(title):
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "inprop": "url",
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page


def main():
    manifest = []
    for title in TITLES:
        page = fetch_article(title)
        if "missing" in page:
            print(f"[WARN] page not found: {title}")
            continue

        extract = page.get("extract", "").strip()
        if not extract:
            print(f"[WARN] empty extract: {title}")
            continue

        slug = slugify(page.get("title", title))
        article_id = f"wikipedia_{slug}"
        record = {
            "article_id": article_id,
            "source_type": "wikipedia",
            "title": page.get("title", title),
            "pageid": page.get("pageid"),
            "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{slug}"),
            "text": extract,
        }

        json_path = OUT_DIR / f"{article_id}.json"
        txt_path = OUT_DIR / f"{article_id}.txt"
        json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        txt_content = f"Title: {record['title']}\nURL: {record['url']}\n\n{record['text']}\n"
        txt_path.write_text(txt_content, encoding="utf-8")

        manifest.append({
            "article_id": article_id,
            "title": record["title"],
            "url": record["url"],
            "chars": len(extract),
        })
        print(f"[OK] {title} -> {len(extract)} chars")
        time.sleep(0.3)

    manifest_path = OUT_DIR / "_wikipedia_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFetched {len(manifest)}/{len(TITLES)} Wikipedia articles.")


if __name__ == "__main__":
    main()
