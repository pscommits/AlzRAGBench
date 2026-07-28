"""Metrics are the one thing that must not be silently wrong — a bug here
corrupts every number in the paper. Hence tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import GenerationMetrics, RetrievalMetrics


class TestRetrievalMetrics:
    def test_recall_perfect(self):
        assert RetrievalMetrics.recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_recall_partial(self):
        assert RetrievalMetrics.recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5

    def test_recall_respects_k(self):
        # 'b' sits at rank 3, so recall@2 must not see it.
        assert RetrievalMetrics.recall_at_k(["a", "x", "b"], ["a", "b"], k=2) == 0.5

    def test_recall_empty_relevant(self):
        assert RetrievalMetrics.recall_at_k(["a"], [], k=1) == 0.0

    def test_mrr_first_position(self):
        assert RetrievalMetrics.mrr(["a", "b"], ["a"]) == 1.0

    def test_mrr_third_position(self):
        assert abs(RetrievalMetrics.mrr(["x", "y", "a"], ["a"]) - 1 / 3) < 1e-9

    def test_mrr_no_hit(self):
        assert RetrievalMetrics.mrr(["x", "y"], ["a"]) == 0.0


class TestGenerationMetrics:
    def test_exact_match_is_normalised(self):
        assert GenerationMetrics.exact_match(["  Lung  "], ["lung"]) == 1.0

    def test_exact_match_negative(self):
        assert GenerationMetrics.exact_match(["heart"], ["lung"]) == 0.0

    def test_token_f1_partial_overlap(self):
        score = GenerationMetrics.token_f1(["left lung opacity"], ["left lung"])
        assert 0.0 < score < 1.0

    def test_token_f1_no_overlap(self):
        assert GenerationMetrics.token_f1(["heart"], ["lung"]) == 0.0
