"""FAISS dense retrieval over BiomedCLIP image-text embeddings."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    def __init__(self, cfg, encoder=None):
        self.cfg = cfg
        self.encoder = encoder
        self.dcfg = cfg.retrieval.dense
        self.index = None
        self.metadata: List[Dict[str, Any]] = []  # index position -> record

    # ---------------- build ----------------
    def build(self, records: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """records[i] must correspond to embeddings[i]. That alignment is the
        whole contract of this class — keep it or retrieval silently lies."""
        import faiss

        if len(records) != embeddings.shape[0]:
            raise ValueError(
                f"records ({len(records)}) and embeddings ({embeddings.shape[0]}) misaligned"
            )

        dim = embeddings.shape[1]
        index_type = self.dcfg.index_type
        logger.info("Building %s index: %d vectors, dim=%d", index_type, len(records), dim)

        if index_type == "IndexFlatIP":
            index = faiss.IndexFlatIP(dim)
        elif index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, self.dcfg.nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.nprobe = self.dcfg.nprobe
        elif index_type == "IndexHNSWFlat":
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unsupported index_type: {index_type}")

        index.add(embeddings)
        self.index = index
        self.metadata = records
        logger.info("Index built: %d vectors", index.ntotal)

    # ---------------- persist ----------------
    def save(self, index_dir: str | Path) -> None:
        import faiss

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "dense.faiss"))
        with (index_dir / "metadata.pkl").open("wb") as f:
            pickle.dump(self.metadata, f)
        with (index_dir / "index_stats.json").open("w") as f:
            json.dump(
                {
                    "n_vectors": int(self.index.ntotal),
                    "index_type": self.dcfg.index_type,
                    "dim": int(self.index.d),
                },
                f,
                indent=2,
            )
        logger.info("Saved index -> %s", index_dir)

    def load(self, index_dir: str | Path) -> None:
        import faiss

        index_dir = Path(index_dir)
        self.index = faiss.read_index(str(index_dir / "dense.faiss"))
        with (index_dir / "metadata.pkl").open("rb") as f:
            self.metadata = pickle.load(f)
        logger.info("Loaded index: %d vectors", self.index.ntotal)

    # ---------------- search ----------------
    def search(self, query_vec: np.ndarray, top_k: int | None = None) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index not built or loaded. Run scripts/build_index.py first.")

        top_k = top_k or self.dcfg.top_k
        q = query_vec.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(q, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS pads with -1 when fewer than top_k exist
                continue
            record = dict(self.metadata[idx])
            record.update({"score": float(score), "rank": rank, "source": "dense"})
            results.append(record)
        return results

    def search_by_text(self, text: str, top_k: int | None = None):
        return self.search(self.encoder.encode_query(text=text), top_k)

    def search_by_image(self, image_path: str, top_k: int | None = None):
        return self.search(self.encoder.encode_query(image_path=image_path), top_k)
