"""Fusing dense and graph evidence into one ranked list for the generator.

The two retrievers return incomparable scores (cosine similarity vs. hop
distance). Fusion is where that's reconciled — get it wrong and one path
silently dominates, which is exactly what the ablation table is there to catch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Evidence:
    """One retrieved item, whatever path it came from."""

    text: str
    source: str                      # "dense" | "graph"
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_id: int | None = None   # assigned post-fusion; used for citations

    def to_prompt_block(self) -> str:
        tag = "IMAGE-TEXT" if self.source == "dense" else "KNOWLEDGE-GRAPH"
        return f"[{self.evidence_id}] ({tag}) {self.text}"


class EvidenceFusion:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fcfg = cfg.retrieval.fusion

    @staticmethod
    def _minmax(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalise each source's scores to [0,1] so they're comparable."""
        if not items:
            return items
        scores = [i["score"] for i in items]
        lo, hi = min(scores), max(scores)
        span = hi - lo
        for item in items:
            item["norm_score"] = 1.0 if span == 0 else (item["score"] - lo) / span
        return items

    def fuse(
        self,
        dense_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
    ) -> List[Evidence]:
        strategy = self.fcfg.strategy
        dense_results = self._minmax(list(dense_results))
        graph_results = self._minmax(list(graph_results))

        if strategy == "weighted":
            merged = self._weighted(dense_results, graph_results)
        elif strategy == "concat":
            merged = dense_results + graph_results
            for m in merged:
                m["fused_score"] = m["norm_score"]
        elif strategy == "rerank":
            raise NotImplementedError(
                "TODO(Nathan): cross-encoder rerank over the union. "
                "Likely the biggest single win available — worth a row in the ablation table."
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

        merged.sort(key=lambda m: -m["fused_score"])

        if self.fcfg.dedupe:
            merged = self._dedupe(merged)

        merged = merged[: self.fcfg.max_evidence_items]

        evidence = []
        for i, m in enumerate(merged, start=1):
            evidence.append(
                Evidence(
                    text=m.get("text") or m.get("caption", ""),
                    source=m["source"],
                    score=m["fused_score"],
                    metadata={
                        k: v
                        for k, v in m.items()
                        if k not in {"text", "caption", "source", "score", "fused_score", "norm_score"}
                    },
                    evidence_id=i,
                )
            )

        logger.debug(
            "Fused %d dense + %d graph -> %d evidence items",
            len(dense_results), len(graph_results), len(evidence),
        )
        return evidence

    def _weighted(self, dense, graph):
        dw, gw = self.fcfg.dense_weight, self.fcfg.graph_weight
        out = []
        for item in dense:
            item["fused_score"] = dw * item["norm_score"]
            out.append(item)
        for item in graph:
            item["fused_score"] = gw * item["norm_score"]
            out.append(item)
        return out

    @staticmethod
    def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Highest-scoring item wins; items arrive pre-sorted."""
        seen, unique = set(), []
        for item in items:
            key = (item.get("text") or item.get("caption", "")).strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(item)
        return unique


def format_evidence_block(evidence: List[Evidence]) -> str:
    """Render evidence for the prompt. Numbered so the model can cite [n]."""
    if not evidence:
        return "No evidence retrieved."
    return "\n".join(e.to_prompt_block() for e in evidence)
