#!/usr/bin/env python
"""Embed every processed record and build the FAISS index.

Usage:
    python scripts/build_index.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding import BiomedCLIPEncoder  # noqa: E402
from src.retrieval import DenseRetriever  # noqa: E402
from src.utils import get_logger, load_config, set_seed  # noqa: E402

logger = get_logger("build_index")


def load_records(processed_dir: Path, datasets: List[str]) -> List[Dict[str, Any]]:
    records = []
    for name in datasets:
        path = processed_dir / name / "pairs.jsonl"
        if not path.exists():
            logger.warning("Missing %s — run scripts/prepare_data.py first", path)
            continue
        with path.open() as f:
            for line in f:
                record = json.loads(line)
                record["dataset"] = name
                records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--modality",
        choices=["image", "text", "both"],
        default="both",
        help="What to embed. 'both' averages the two — see BiomedCLIPEncoder.encode_query.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)

    enabled = [n for n, d in cfg.datasets.items() if d.get("enabled")]
    records = load_records(Path(cfg.paths.processed_dir), enabled)
    if not records:
        logger.error("No records found. Nothing to index.")
        return

    logger.info("Embedding %d records (mode=%s)", len(records), args.modality)
    encoder = BiomedCLIPEncoder(cfg)

    if args.modality == "image":
        embeddings = encoder.encode_images([r["image_path"] for r in records])
    elif args.modality == "text":
        embeddings = encoder.encode_texts([r.get("caption") or r.get("question", "") for r in records])
    else:
        import numpy as np

        img = encoder.encode_images([r["image_path"] for r in records])
        txt = encoder.encode_texts([r.get("caption") or r.get("question", "") for r in records])
        embeddings = (img + txt) / 2
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings.astype("float32")

    retriever = DenseRetriever(cfg, encoder=encoder)
    retriever.build(records, embeddings)
    retriever.save(cfg.paths.index_dir)


if __name__ == "__main__":
    main()
