"""MedGemma generation over fused evidence."""
from __future__ import annotations

from typing import List

from src.retrieval.fusion import Evidence
from src.utils import get_logger

from .prompts import NO_RETRIEVAL_PROMPT, build_prompt

logger = get_logger(__name__)


class MedGemmaGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gcfg = cfg.models.generation
        self.device = cfg.experiment.device
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_id = self.gcfg.hf_id
        logger.info("Loading generator: %s", hf_id)

        self._tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=getattr(torch, self.gcfg.dtype, torch.float32),
            device_map="auto" if self.device == "cuda" else None,
        ).eval()

    def generate(self, question: str, evidence: List[Evidence] | None = None) -> str:
        """evidence=None runs the no-retrieval ablation baseline."""
        self._ensure_loaded()
        import torch

        prompt = (
            build_prompt(self.cfg, question, evidence)
            if evidence
            else NO_RETRIEVAL_PROMPT.format(question=question)
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.gcfg.max_new_tokens,
                temperature=self.gcfg.temperature,
                top_p=self.gcfg.top_p,
                do_sample=self.gcfg.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Slice off the prompt so we return only the completion.
        completion = output[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(completion, skip_special_tokens=True).strip()
