"""BiomedCLIP wrapper — the single place image and text become vectors.

Both modalities land in the same space, which is what lets the FAISS index
answer image->text and text->image with one index instead of two.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image

from src.utils import get_logger

logger = get_logger(__name__)


class BiomedCLIPEncoder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.experiment.device
        self.embed_dim = cfg.models.embedding.embed_dim
        self.normalize = cfg.models.embedding.normalize
        self.batch_size = cfg.models.embedding.batch_size
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    # -- lazy load: keeps import cheap and lets tests run without weights --
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import open_clip

        hf_id = self.cfg.models.embedding.hf_id
        logger.info("Loading BiomedCLIP: %s", hf_id)

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            f"hf-hub:{hf_id}"
        )
        self._tokenizer = open_clip.get_tokenizer(f"hf-hub:{hf_id}")
        self._model = self._model.to(self.device).eval()

    def _post(self, feats: torch.Tensor) -> np.ndarray:
        if self.normalize:
            # L2-normalise so inner product == cosine similarity in FAISS
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    @torch.no_grad()
    def encode_images(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        """-> (N, embed_dim) float32"""
        self._ensure_loaded()
        out: List[np.ndarray] = []

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            tensors = [
                self._preprocess(Image.open(p).convert("RGB")) for p in batch_paths
            ]
            batch = torch.stack(tensors).to(self.device)
            out.append(self._post(self._model.encode_image(batch)))

        return np.vstack(out) if out else np.empty((0, self.embed_dim), dtype="float32")

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """-> (N, embed_dim) float32"""
        self._ensure_loaded()
        out: List[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = self._tokenizer(list(texts[i : i + self.batch_size])).to(self.device)
            out.append(self._post(self._model.encode_text(batch)))

        return np.vstack(out) if out else np.empty((0, self.embed_dim), dtype="float32")

    def encode_query(self, image_path=None, text=None) -> np.ndarray:
        """Encode a query that may be image, text, or both.

        When both are given they are averaged — a deliberately simple fusion.
        TODO(Nathan): ablate this against concatenation / learned fusion.
        """
        parts = []
        if image_path is not None:
            parts.append(self.encode_images([image_path])[0])
        if text is not None:
            parts.append(self.encode_texts([text])[0])

        if not parts:
            raise ValueError("encode_query needs at least one of image_path, text")

        vec = np.mean(parts, axis=0)
        if self.normalize:
            vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.astype("float32")
