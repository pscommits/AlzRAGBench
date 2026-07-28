"""End-to-end orchestration.

Both the CLI and the Hugging Face Space call this class. That is deliberate:
the demo and the reported numbers run identical code, so they cannot drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from src.embedding import BiomedCLIPEncoder
from src.evaluation import HallucinationScorer
from src.generation import MedGemmaGenerator
from src.retrieval import DenseRetriever, EvidenceFusion, Evidence, KnowledgeGraphRetriever
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class RAGOutput:
    answer: str
    evidence: List[Evidence] = field(default_factory=list)
    hallucination: Dict[str, Any] | None = None
    question: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "evidence": [
                {
                    "id": e.evidence_id,
                    "text": e.text,
                    "source": e.source,
                    "score": round(e.score, 4),
                }
                for e in self.evidence
            ],
            "hallucination": self.hallucination,
        }


class KGMMRAGPipeline:
    def __init__(self, cfg, load_artifacts: bool = True):
        self.cfg = cfg

        self.encoder = BiomedCLIPEncoder(cfg)
        self.dense = DenseRetriever(cfg, encoder=self.encoder)
        self.graph = KnowledgeGraphRetriever(cfg, encoder=self.encoder)
        self.fusion = EvidenceFusion(cfg)
        self.generator = MedGemmaGenerator(cfg)
        self.hallucination = HallucinationScorer(cfg)

        if load_artifacts:
            self._load_artifacts()

    def _load_artifacts(self) -> None:
        index_dir = Path(self.cfg.paths.index_dir)
        graph_dir = Path(self.cfg.paths.graph_dir)

        if (index_dir / "dense.faiss").exists():
            self.dense.load(index_dir)
        else:
            logger.warning("No FAISS index at %s — run scripts/build_index.py", index_dir)

        if (graph_dir / "kg.pkl").exists():
            self.graph.load(graph_dir)
        else:
            logger.warning("No KG at %s — run scripts/build_kg.py", graph_dir)

    def retrieve(
        self,
        question: str,
        image_path: str | None = None,
        use_dense: bool = True,
        use_graph: bool = True,
    ) -> List[Evidence]:
        """Flags exist so the ablation table (no_retrieval / dense_only /
        graph_only / fused) runs through the same code path as the real system."""
        dense_results: List[Dict] = []
        graph_results: List[Dict] = []

        if use_dense and self.dense.index is not None:
            query_vec = self.encoder.encode_query(image_path=image_path, text=question)
            dense_results = self.dense.search(query_vec)

        if use_graph and self.graph.graph.number_of_nodes() > 0:
            graph_results = self.graph.retrieve(question)

        return self.fusion.fuse(dense_results, graph_results)

    def run(
        self,
        question: str,
        image_path: str | None = None,
        use_dense: bool = True,
        use_graph: bool = True,
        score_hallucination: bool = True,
    ) -> RAGOutput:
        evidence = (
            self.retrieve(question, image_path, use_dense, use_graph)
            if (use_dense or use_graph)
            else []
        )

        answer = self.generator.generate(question, evidence if evidence else None)

        hallucination = None
        if score_hallucination:
            hallucination = self.hallucination.score(answer, evidence).to_dict()

        return RAGOutput(
            answer=answer,
            evidence=evidence,
            hallucination=hallucination,
            question=question,
        )
