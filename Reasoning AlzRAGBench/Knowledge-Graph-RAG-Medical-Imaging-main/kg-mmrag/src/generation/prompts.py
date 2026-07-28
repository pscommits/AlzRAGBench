"""Prompt construction. Kept separate from the model so prompts can be
ablated without touching inference code."""
from __future__ import annotations

from typing import List

from src.retrieval.fusion import Evidence, format_evidence_block


def build_prompt(cfg, question: str, evidence: List[Evidence]) -> str:
    system = cfg.prompting.system.strip()
    evidence_block = format_evidence_block(evidence)

    citation_hint = (
        "Cite the evidence number(s) supporting each claim, e.g. [2]."
        if cfg.prompting.include_evidence_ids
        else ""
    )

    return f"""{system}

EVIDENCE
--------
{evidence_block}

QUESTION
--------
{question}

INSTRUCTIONS
------------
Answer using only the evidence above. {citation_hint}
If the evidence does not support an answer, state that explicitly instead of guessing.

ANSWER:"""


NO_RETRIEVAL_PROMPT = """You are a medical imaging assistant. Answer the question directly.

QUESTION
--------
{question}

ANSWER:"""
