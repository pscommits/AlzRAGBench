"""
Chunk every article in 'Selected articles/' into overlapping word-windows
suitable for vector embedding. Uses each article's own JSON record (not the
.txt mirror) so metadata (title, source_type, url, etc.) rides along with
every chunk.

Chunking strategy: fixed-size word windows (~220 words, ~40-word overlap)
approximating ~300 token chunks for typical embedding models, with sentence
boundary snapping so chunks don't cut mid-sentence.
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "Selected articles"
OUT_DIR = BASE_DIR / "chunking"
OUT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE_WORDS = 220
OVERLAP_WORDS = 40

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def get_article_text(record):
    if record["source_type"] == "pubmed":
        parts = [f"Title: {record['title']}"]
        if record.get("abstract"):
            parts.append(record["abstract"])
        return "\n\n".join(parts)
    # wikipedia + textbook records already store a unified 'text' field
    return f"Title: {record['title']}\n\n{record.get('text', '')}"


def sentence_chunks(text, chunk_size_words, overlap_words):
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    chunks = []
    current_words = []
    current_sentences = []

    for sent in sentences:
        sent_words = sent.split()
        if current_words and len(current_words) + len(sent_words) > chunk_size_words:
            chunks.append(" ".join(current_sentences).strip())
            # start next chunk with overlap: carry trailing words as context
            overlap_text = " ".join(current_words[-overlap_words:])
            current_words = overlap_text.split()
            current_sentences = [overlap_text] if overlap_text else []
        current_sentences.append(sent)
        current_words.extend(sent_words)

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return [c for c in chunks if c]


def main():
    manifest = []
    total_chunks = 0

    json_files = sorted(
        p for p in SRC_DIR.glob("*.json") if not p.name.startswith("_")
    )

    for jf in json_files:
        record = json.loads(jf.read_text(encoding="utf-8"))
        article_id = record["article_id"]
        text = get_article_text(record)
        chunks = sentence_chunks(text, CHUNK_SIZE_WORDS, OVERLAP_WORDS)

        article_chunks = []
        offset = 0
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{article_id}__chunk{i:03d}"
            start = text.find(chunk_text[:50], offset) if chunk_text else -1
            char_start = start if start != -1 else None
            char_end = (char_start + len(chunk_text)) if char_start is not None else None
            if char_start is not None:
                offset = char_start

            chunk_record = {
                "chunk_id": chunk_id,
                "article_id": article_id,
                "source_type": record["source_type"],
                "title": record["title"],
                "chunk_index": i,
                "num_chunks_in_article": len(chunks),
                "word_count": len(chunk_text.split()),
                "char_start": char_start,
                "char_end": char_end,
                "text": chunk_text,
                "metadata": {
                    k: v for k, v in record.items()
                    if k not in ("text", "abstract", "sections")
                },
            }
            article_chunks.append(chunk_record)

        out_path = OUT_DIR / f"{article_id}__chunks.json"
        out_path.write_text(json.dumps(article_chunks, indent=2, ensure_ascii=False), encoding="utf-8")

        manifest.append({
            "article_id": article_id,
            "source_type": record["source_type"],
            "num_chunks": len(chunks),
            "chunks_file": out_path.name,
        })
        total_chunks += len(chunks)
        print(f"[OK] {article_id}: {len(chunks)} chunks")

    # Also emit one flat file with every chunk (handy for building a single
    # vector index across the whole corpus).
    all_chunks = []
    for jf in sorted(OUT_DIR.glob("*__chunks.json")):
        all_chunks.extend(json.loads(jf.read_text(encoding="utf-8")))

    (OUT_DIR / "_all_chunks.json").write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "_chunking_manifest.json").write_text(
        json.dumps({
            "chunk_size_words": CHUNK_SIZE_WORDS,
            "overlap_words": OVERLAP_WORDS,
            "num_articles": len(manifest),
            "total_chunks": total_chunks,
            "articles": manifest,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nTotal: {len(manifest)} articles -> {total_chunks} chunks")


if __name__ == "__main__":
    main()
