#!/usr/bin/env python
"""Single-query CLI — useful for eyeballing evidence traces.

Usage:
    python scripts/query.py --question "What abnormality is visible?" --image scan.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import KGMMRAGPipeline  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--question", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)

    pipeline = KGMMRAGPipeline(cfg)
    out = pipeline.run(question=args.question, image_path=args.image)

    if args.json:
        print(json.dumps(out.to_dict(), indent=2))
        return

    print("\n" + "=" * 70)
    print(f"Q: {out.question}")
    print("=" * 70)
    print(f"\nANSWER\n{out.answer}\n")

    print("EVIDENCE")
    for e in out.evidence:
        print(f"  [{e.evidence_id}] ({e.source}, {e.score:.3f}) {e.text[:100]}")

    if out.hallucination:
        h = out.hallucination
        print(f"\nGROUNDING: {h['n_grounded']}/{h['n_claims']} claims supported "
              f"(hallucination rate {h['hallucination_rate']:.2%})")
        for c in h["claims"]:
            mark = "OK  " if c["grounded"] else "??  "
            src = f"-> [{c['evidence_id']}]" if c["evidence_id"] else "-> ungrounded"
            print(f"  {mark}{c['claim'][:70]} {src}")
    print()


if __name__ == "__main__":
    main()
