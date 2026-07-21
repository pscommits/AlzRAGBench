"""
Fetch the open-access StatPearls 'Alzheimer Disease' chapter from NCBI Bookshelf
(NBK499922) as the second VectorRAG source (textbook-style long-form content).
StatPearls content is produced under a CC/public-reuse license via StatPearls
Publishing and is free to redistribute for research/education use.
Saves JSON + plain-text into 'Selected articles/'.
"""
import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path(__file__).resolve().parent.parent / "Selected articles"
OUT_DIR.mkdir(exist_ok=True)

NBK_ID = "NBK499922"
URL = f"https://www.ncbi.nlm.nih.gov/books/{NBK_ID}/"
HEADERS = {"User-Agent": "HybridRAG-dataset-builder/1.0 (research use; contact: prabhat0405005@gmail.com)"}


def main():
    r = requests.get(URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    article = soup.find("div", class_="article") or soup.find("div", id="maincontent") or soup

    # Strip nav/aside/script/style clutter typical of Bookshelf pages.
    for tag in article.find_all(["script", "style", "nav", "aside"]):
        tag.decompose()

    sections = []
    current_heading = "Introduction"
    buf = []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append((current_heading, text))

    for el in article.find_all(["h2", "h3", "p", "li"]):
        if el.name in ("h2", "h3"):
            flush()
            current_heading = el.get_text(strip=True)
            buf = []
        else:
            txt = el.get_text(" ", strip=True)
            if txt:
                buf.append(txt)
    flush()

    if not sections:
        raise RuntimeError("Could not parse any sections from the Bookshelf page — page structure may differ")

    full_text_parts = [f"## {heading}\n{text}" for heading, text in sections]
    full_text = "\n\n".join(full_text_parts)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

    if soup.title and soup.title.get_text():
        title = soup.title.get_text().split(" - ")[0].strip()
    else:
        h1s = soup.find_all("h1")
        title = h1s[-1].get_text(strip=True) if h1s else "Alzheimer Disease"

    record = {
        "article_id": "textbook_statpearls_alzheimer_disease",
        "source_type": "textbook",
        "title": title,
        "book": "StatPearls",
        "accession": NBK_ID,
        "url": URL,
        "license": "StatPearls Publishing content is freely reusable (public-reuse license via NCBI Bookshelf)",
        "sections": [{"heading": h, "text": t} for h, t in sections],
        "text": full_text,
    }

    json_path = OUT_DIR / "textbook_statpearls_alzheimer_disease.json"
    txt_path = OUT_DIR / "textbook_statpearls_alzheimer_disease.txt"
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_content = f"Title: {title}\nBook: StatPearls\nURL: {URL}\n\n{full_text}\n"
    txt_path.write_text(txt_content, encoding="utf-8")

    print(f"[OK] {title} -> {len(sections)} sections, {len(full_text)} chars")


if __name__ == "__main__":
    main()
