"""Config inheritance is load-bearing — eval.yaml inherits default.yaml, and a
silent merge bug would mean evaluating a different system than you think."""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import Config, _deep_merge, load_config


def test_attribute_access():
    cfg = Config({"models": {"embedding": {"embed_dim": 512}}})
    assert cfg.models.embedding.embed_dim == 512


def test_deep_merge_overrides_leaf_only():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 99}}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 99}, "b": 3}


def test_inherits_chain(tmp_path):
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"

    parent.write_text(yaml.safe_dump({"retrieval": {"dense": {"top_k": 10}}, "seed": 42}))
    child.write_text(
        yaml.safe_dump({"inherits": str(parent), "retrieval": {"dense": {"top_k": 5}}})
    )

    cfg = load_config(child)
    assert cfg.retrieval.dense.top_k == 5   # child wins
    assert cfg.seed == 42                    # parent survives
