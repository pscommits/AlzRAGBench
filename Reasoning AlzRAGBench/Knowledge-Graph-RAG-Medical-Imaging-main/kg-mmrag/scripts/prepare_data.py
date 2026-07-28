#!/usr/bin/env python
"""Normalise every dataset into one schema.

Downstream code never needs to know whether a record came from SLAKE, BraTS,
or ROCOv2 — that knowledge is quarantined here.

Usage:
    python scripts/prepare_data.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, load_config, set_seed  # noqa: E402

logger = get_logger("prepare_data")

UNIFIED_SCHEMA = {
    "id": str,
    "image_path": str,
    "modality": str,
    "caption": str,
    "question": str,
    "answer": str,
    "entities": list,
    "split": str,
}


def prepare_slake(cfg, dataset_cfg) -> List[Dict[str, Any]]:
    """TODO(Nathan): implement. SLAKE ships a knowledge base alongside the VQA
    pairs — emit its triples to processed/slake/triples.jsonl for build_kg.py."""
    raise NotImplementedError("prepare_slake")


def prepare_brats(cfg, dataset_cfg) -> List[Dict[str, Any]]:
    """TODO(Nathan): implement. BraTS is volumetric — slice to 2D and record
    which slice index each record came from so results stay traceable."""
    raise NotImplementedError("prepare_brats")


def prepare_rocov2(cfg, dataset_cfg) -> List[Dict[str, Any]]:
    """TODO(Nathan): implement. ROCOv2 carries concept annotations — those are
    KG entity candidates, don't discard them."""
    raise NotImplementedError("prepare_rocov2")


PREPARERS = {
    "slake": prepare_slake,
    "brats": prepare_brats,
    "rocov2": prepare_rocov2,
}


def write_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    logger.info("Wrote %d records -> %s", len(records), path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", nargs="*", help="Subset to prepare (default: all enabled)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)

    processed_dir = Path(cfg.paths.processed_dir)
    targets = args.datasets or [
        name for name, dcfg in cfg.datasets.items() if dcfg.get("enabled", False)
    ]

    manifest = {}
    for name in targets:
        if name not in PREPARERS:
            logger.warning("No preparer for '%s' — skipping", name)
            continue

        logger.info("Preparing %s ...", name)
        records = PREPARERS[name](cfg, cfg.datasets[name])
        write_jsonl(records, processed_dir / name / "pairs.jsonl")

        manifest[name] = {
            "n_records": len(records),
            "modality": cfg.datasets[name]["modality"],
            "splits": sorted({r["split"] for r in records}),
        }

    with (processed_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Done. Manifest: %s", manifest)


if __name__ == "__main__":
    main()
