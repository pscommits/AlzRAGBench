"""Retrieval and generation metrics.

Every number in the README results table comes from here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


class RetrievalMetrics:
    """Standard IR metrics. `retrieved` is a ranked list of ids;
    `relevant` is the set of ground-truth ids."""

    @staticmethod
    def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
        if not relevant:
            return 0.0
        hits = len(set(retrieved[:k]) & set(relevant))
        return hits / len(set(relevant))

    @staticmethod
    def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
        if k == 0:
            return 0.0
        return len(set(retrieved[:k]) & set(relevant)) / k

    @staticmethod
    def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
        """Reciprocal rank of the first relevant hit."""
        relevant_set = set(relevant)
        for rank, item in enumerate(retrieved, start=1):
            if item in relevant_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
        relevant_set = set(relevant)
        gains = [1.0 if item in relevant_set else 0.0 for item in retrieved[:k]]
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))
        return dcg / ideal if ideal > 0 else 0.0

    @classmethod
    def compute_all(
        cls,
        results: List[Dict[str, Any]],
        k_values: Sequence[int] = (1, 5, 10),
    ) -> Dict[str, float]:
        """results: [{"retrieved": [ids...], "relevant": [ids...]}, ...]"""
        out: Dict[str, float] = {}
        for k in k_values:
            out[f"recall@{k}"] = float(
                np.mean([cls.recall_at_k(r["retrieved"], r["relevant"], k) for r in results])
            )
            out[f"precision@{k}"] = float(
                np.mean([cls.precision_at_k(r["retrieved"], r["relevant"], k) for r in results])
            )
        out["mrr"] = float(np.mean([cls.mrr(r["retrieved"], r["relevant"]) for r in results]))
        out["ndcg@10"] = float(
            np.mean([cls.ndcg_at_k(r["retrieved"], r["relevant"], 10) for r in results])
        )
        return {k: round(v, 4) for k, v in out.items()}


class GenerationMetrics:
    """VQA accuracy + text overlap. For closed-form VQA, exact-match accuracy
    is the number that matters; ROUGE/BERTScore are for open-ended captions."""

    @staticmethod
    def _norm(s: str) -> str:
        return " ".join(s.lower().strip().split())

    @classmethod
    def exact_match(cls, predictions: Sequence[str], references: Sequence[str]) -> float:
        return float(
            np.mean([cls._norm(p) == cls._norm(r) for p, r in zip(predictions, references)])
        )

    @classmethod
    def token_f1(cls, predictions: Sequence[str], references: Sequence[str]) -> float:
        """Token-overlap F1 — tolerates phrasing differences that exact-match punishes."""
        scores = []
        for pred, ref in zip(predictions, references):
            p_tokens, r_tokens = set(cls._norm(pred).split()), set(cls._norm(ref).split())
            if not p_tokens or not r_tokens:
                scores.append(0.0)
                continue
            overlap = len(p_tokens & r_tokens)
            if overlap == 0:
                scores.append(0.0)
                continue
            precision = overlap / len(p_tokens)
            recall = overlap / len(r_tokens)
            scores.append(2 * precision * recall / (precision + recall))
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def bertscore(predictions: Sequence[str], references: Sequence[str]) -> float:
        from bert_score import score as bert_score_fn

        _, _, f1 = bert_score_fn(list(predictions), list(references), lang="en", verbose=False)
        return float(f1.mean())

    @classmethod
    def compute_all(
        cls,
        predictions: Sequence[str],
        references: Sequence[str],
        metrics: Sequence[str] = ("accuracy", "f1", "rouge_l"),
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if "accuracy" in metrics:
            out["accuracy"] = cls.exact_match(predictions, references)
        if "f1" in metrics:
            out["f1"] = cls.token_f1(predictions, references)
        if "rouge_l" in metrics:
            out["rouge_l"] = cls.rouge_l(predictions, references)
        if "bertscore" in metrics:
            out["bertscore"] = cls.bertscore(predictions, references)
        return {k: round(v, 4) for k, v in out.items()}
