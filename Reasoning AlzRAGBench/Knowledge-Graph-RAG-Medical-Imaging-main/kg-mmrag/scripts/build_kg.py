#!/usr/bin/env python
"""Construct the clinical knowledge graph from dataset annotations.

Usage:
    python scripts/build_kg.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import KnowledgeGraphRetriever  # noqa: E402
from src.utils import get_logger, load_config, set_seed  # noqa: E402

logger = get_logger("build_kg")


def load_triples(processed_dir: Path, datasets: List[str]) -> List[Dict[str, str]]:
    """Each dataset that carries structured annotations should emit
    triples.jsonl during prepare_data.py.

    TODO(Nathan): SLAKE's bundled KB is the main source here. If you are also
    mapping to UMLS/RadLex, note the licence in data/README.md.
    """
    triples: List[Dict[str, str]] = []
    for name in datasets:
        path = processed_dir / name / "triples.jsonl"
        if not path.exists():
            logger.warning("No triples for '%s' at %s — skipping", name, path)
            continue
        with path.open() as f:
            for line in f:
                triple = json.loads(line)
                triple.setdefault("source", name)
                triples.append(triple)
    return triples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)

    enabled = [n for n, d in cfg.datasets.items() if d.get("enabled")]
    triples = load_triples(Path(cfg.paths.processed_dir), enabled)

    if not triples:
        logger.error("No triples found. Cannot build KG.")
        return

    kg = KnowledgeGraphRetriever(cfg)
    kg.build(triples)
    kg.save(cfg.paths.graph_dir)

    logger.info("Copy the stats from artifacts/graphs/stats.json into data/README.md")


if __name__ == "__main__":
    main()
