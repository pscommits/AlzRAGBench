"""Hallucination scoring — the differentiating contribution of this system.

Decompose the generated answer into atomic claims, then check each claim
against the retrieved evidence. A claim entailed by no evidence item is
ungrounded. The hallucination rate is the fraction of such claims.

This is what makes the KG worth its complexity: the claim is not merely that
fused retrieval scores higher, but that it hallucinates *less*, and this
module is what substantiates that.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.retrieval.fusion import Evidence
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ClaimScore:
    claim: str
    grounded: bool
    max_entailment: float
    supporting_evidence_id: int | None


@dataclass
class HallucinationResult:
    hallucination_rate: float          # fraction of claims that are ungrounded
    n_claims: int
    n_grounded: int
    claim_scores: List[ClaimScore]

    def to_dict(self) -> Dict:
        return {
            "hallucination_rate": round(self.hallucination_rate, 4),
            "n_claims": self.n_claims,
            "n_grounded": self.n_grounded,
            "claims": [
                {
                    "claim": c.claim,
                    "grounded": c.grounded,
                    "entailment": round(c.max_entailment, 4),
                    "evidence_id": c.supporting_evidence_id,
                }
                for c in self.claim_scores
            ],
        }


class HallucinationScorer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hcfg = cfg.models.hallucination
        self.threshold = self.hcfg.threshold
        self._nli = None

    def _ensure_loaded(self) -> None:
        if self._nli is not None:
            return
        from transformers import pipeline

        logger.info("Loading entailment model: %s", self.hcfg.hf_id)
        self._nli = pipeline(
            "text-classification",
            model=self.hcfg.hf_id,
            device=0 if self.cfg.experiment.device == "cuda" else -1,
        )

    # ---------------- claim decomposition ----------------
    @staticmethod
    def extract_claims(answer: str) -> List[str]:
        """Split an answer into atomic claims.

        Sentence-splitting is a crude proxy — a sentence can carry two claims.
        TODO(Nathan): compare against LLM-based claim decomposition; the
        current approach probably *under*-reports hallucination, which is the
        conservative direction but worth quantifying.
        """
        stripped = re.sub(r"\[\d+\]", "", answer)  # drop citation markers
        sentences = re.split(r"(?<=[.!?])\s+", stripped)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    # ---------------- entailment ----------------
    def _entailment(self, premise: str, hypothesis: str) -> float:
        self._ensure_loaded()
        result = self._nli(f"{premise} [SEP] {hypothesis}", truncation=True)[0]
        label = result["label"].lower()
        if "entail" in label:
            return float(result["score"])
        if "neutral" in label:
            return float(result["score"]) * 0.5
        return 1.0 - float(result["score"])  # contradiction

    def score(self, answer: str, evidence: List[Evidence]) -> HallucinationResult:
        claims = self.extract_claims(answer)

        if not claims:
            return HallucinationResult(0.0, 0, 0, [])

        if not evidence:
            # No evidence retrieved -> every claim is unsupported by construction.
            scores = [ClaimScore(c, False, 0.0, None) for c in claims]
            return HallucinationResult(1.0, len(claims), 0, scores)

        claim_scores: List[ClaimScore] = []
        for claim in claims:
            best_score, best_id = 0.0, None
            for ev in evidence:
                s = self._entailment(ev.text, claim)
                if s > best_score:
                    best_score, best_id = s, ev.evidence_id

            grounded = best_score >= self.threshold
            claim_scores.append(
                ClaimScore(claim, grounded, best_score, best_id if grounded else None)
            )

        n_grounded = sum(c.grounded for c in claim_scores)
        rate = 1.0 - (n_grounded / len(claims))

        return HallucinationResult(rate, len(claims), n_grounded, claim_scores)

    def score_batch(self, answers: List[str], evidences: List[List[Evidence]]) -> Dict:
        results = [self.score(a, e) for a, e in zip(answers, evidences)]
        return {
            "mean_hallucination_rate": round(
                float(np.mean([r.hallucination_rate for r in results])), 4
            ),
            "total_claims": sum(r.n_claims for r in results),
            "total_grounded": sum(r.n_grounded for r in results),
            "per_sample": [r.to_dict() for r in results],
        }
