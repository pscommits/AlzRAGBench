#!/usr/bin/env python
"""Reproduce the README results table.

Runs the full ablation grid (no_retrieval / dense_only / graph_only / fused)
and writes results/metrics.json. If the README table and this script's output
disagree, the README is wrong.

Usage:
    python scripts/run_eval.py --config configs/eval.yaml
    python scripts/run_eval.py --config configs/eval.yaml --ablations --n 100
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import GenerationMetrics, RetrievalMetrics  # noqa: E402
from src.pipeline import KGMMRAGPipeline  # noqa: E402
from src.utils import get_logger, load_config, set_seed  # noqa: E402
from src.utils.config import save_config  # noqa: E402

logger = get_logger("run_eval")

ABLATIONS = {
    "no_retrieval": dict(use_dense=False, use_graph=False),
    "dense_only": dict(use_dense=True, use_graph=False),
    "graph_only": dict(use_dense=False, use_graph=True),
    "fused": dict(use_dense=True, use_graph=True),
}


def load_eval_set(cfg, split: str, n: int | None) -> List[Dict[str, Any]]:
    processed = Path(cfg.paths.processed_dir)
    samples = []
    for name, dcfg in cfg.datasets.items():
        if not dcfg.get("enabled"):
            continue
        path = processed / name / "pairs.jsonl"
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                record = json.loads(line)
                if record.get("split") == split:
                    record["dataset"] = name
                    samples.append(record)
    return samples[:n] if n else samples


def evaluate(pipeline, samples, cfg, ablation: str) -> Dict[str, Any]:
    from tqdm import tqdm

    flags = ABLATIONS[ablation]
    predictions, references, retrieval_pairs, hallucination_rates = [], [], [], []

    for sample in tqdm(samples, desc=ablation):
        out = pipeline.run(
            question=sample["question"],
            image_path=sample.get("image_path"),
            score_hallucination=cfg.evaluation.hallucination.enabled,
            **flags,
        )

        predictions.append(out.answer)
        references.append(sample["answer"])

        if out.hallucination:
            hallucination_rates.append(out.hallucination["hallucination_rate"])

        if flags["use_dense"] or flags["use_graph"]:
            retrieval_pairs.append(
                {
                    "retrieved": [e.metadata.get("id", "") for e in out.evidence],
                    "relevant": [sample["id"]],
                }
            )

    result: Dict[str, Any] = {
        "n_samples": len(samples),
        "generation": GenerationMetrics.compute_all(
            predictions, references, cfg.evaluation.generation_metrics
        ),
    }

    if retrieval_pairs:
        result["retrieval"] = RetrievalMetrics.compute_all(retrieval_pairs)

    if hallucination_rates:
        result["hallucination_rate"] = round(
            sum(hallucination_rates) / len(hallucination_rates), 4
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--ablations", action="store_true", help="Run the full ablation grid")
    parser.add_argument("--n", type=int, default=None, help="Limit samples (smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)

    samples = load_eval_set(cfg, cfg.evaluation.split, args.n or cfg.evaluation.n_samples)
    if not samples:
        logger.error("No eval samples found for split '%s'", cfg.evaluation.split)
        return

    logger.info("Evaluating on %d samples", len(samples))
    pipeline = KGMMRAGPipeline(cfg)

    to_run = list(ABLATIONS) if args.ablations else ["fused"]
    results = {name: evaluate(pipeline, samples, cfg, name) for name in to_run}

    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": args.config,
            "split": cfg.evaluation.split,
            "n_samples": len(samples),
            "seed": cfg.experiment.seed,
            "embedding_model": cfg.models.embedding.hf_id,
            "generation_model": cfg.models.generation.hf_id,
        },
        "results": results,
    }

    out_path = Path(args.out or cfg.output.metrics_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    # Resolved config lands beside the metrics, so a number can always be
    # traced back to the exact settings that produced it.
    save_config(cfg, out_path.parent / "resolved_config.yaml")

    logger.info("Wrote %s", out_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
