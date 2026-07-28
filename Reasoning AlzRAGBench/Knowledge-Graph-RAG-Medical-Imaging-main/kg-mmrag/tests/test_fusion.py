"""Fusion reconciles two incomparable score scales. If normalisation breaks,
one retrieval path silently dominates and the ablation table becomes a lie."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.fusion import EvidenceFusion
from src.utils.config import Config

CFG = Config(
    {
        "retrieval": {
            "fusion": {
                "strategy": "weighted",
                "dense_weight": 0.6,
                "graph_weight": 0.4,
                "max_evidence_items": 8,
                "dedupe": True,
            }
        }
    }
)


def _dense(text, score):
    return {"text": text, "score": score, "source": "dense"}


def _graph(text, score):
    return {"text": text, "score": score, "source": "graph"}


def test_fusion_assigns_sequential_ids():
    fusion = EvidenceFusion(CFG)
    ev = fusion.fuse([_dense("a", 0.9), _dense("b", 0.5)], [_graph("c", 1.0)])
    assert [e.evidence_id for e in ev] == list(range(1, len(ev) + 1))


def test_fusion_dedupes_identical_text():
    fusion = EvidenceFusion(CFG)
    ev = fusion.fuse([_dense("same finding", 0.9)], [_graph("same finding", 1.0)])
    assert len(ev) == 1


def test_fusion_respects_max_items():
    fusion = EvidenceFusion(CFG)
    dense = [_dense(f"d{i}", 0.9 - i * 0.01) for i in range(20)]
    ev = fusion.fuse(dense, [])
    assert len(ev) <= CFG.retrieval.fusion.max_evidence_items


def test_fusion_handles_empty_graph_path():
    """The dense_only ablation passes an empty graph list — must not crash."""
    fusion = EvidenceFusion(CFG)
    ev = fusion.fuse([_dense("a", 0.9)], [])
    assert len(ev) == 1 and ev[0].source == "dense"


def test_fusion_handles_both_empty():
    fusion = EvidenceFusion(CFG)
    assert fusion.fuse([], []) == []
