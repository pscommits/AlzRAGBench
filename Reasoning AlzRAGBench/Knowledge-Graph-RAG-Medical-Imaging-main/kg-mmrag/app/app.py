"""Gradio front-end for the Hugging Face Space.

Deliberately thin: it calls the same KGMMRAGPipeline the evaluation harness
calls. The demo and the reported numbers therefore cannot diverge.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

from src.pipeline import KGMMRAGPipeline  # noqa: E402
from src.utils import load_config  # noqa: E402

cfg = load_config("configs/default.yaml")
pipeline = KGMMRAGPipeline(cfg)


def _format_evidence(evidence) -> str:
    if not evidence:
        return "_No evidence retrieved._"
    lines = []
    for e in evidence:
        tag = "Image-Text" if e.source == "dense" else "Knowledge Graph"
        lines.append(f"**[{e.evidence_id}]** _{tag}_ · `{e.score:.3f}`  \n{e.text}")
    return "\n\n".join(lines)


def _format_grounding(h) -> str:
    if not h:
        return ""
    rate = h["hallucination_rate"]
    verdict = "Well grounded" if rate < 0.2 else "Partially grounded" if rate < 0.5 else "Weakly grounded"
    lines = [
        f"### {verdict}",
        f"{h['n_grounded']} of {h['n_claims']} claims supported by retrieved evidence "
        f"(hallucination rate **{rate:.1%}**)\n",
    ]
    for c in h["claims"]:
        mark = "✅" if c["grounded"] else "⚠️"
        ref = f" → evidence [{c['evidence_id']}]" if c["evidence_id"] else " → **unsupported**"
        lines.append(f"{mark} {c['claim']}{ref}")
    return "\n".join(lines)


def answer(image, question, use_dense, use_graph):
    if not question or not question.strip():
        return "Please enter a question.", "", ""

    out = pipeline.run(
        question=question,
        image_path=image,
        use_dense=use_dense,
        use_graph=use_graph,
    )
    return out.answer, _format_evidence(out.evidence), _format_grounding(out.hallucination)


with gr.Blocks(title="KG-MMRAG") as demo:
    gr.Markdown(
        """
        # KG-MMRAG
        **Knowledge Graph–Augmented Multimodal RAG for Medical Imaging**

        Answers are grounded in two retrieval paths — a dense image–text index and a
        clinical knowledge graph. Every claim in the answer is checked against the
        retrieved evidence and flagged if unsupported.

        > ⚠️ Research prototype. Not a medical device. Not for clinical use.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="filepath", label="Medical image (optional)")
            question_in = gr.Textbox(
                label="Question",
                placeholder="What abnormality is visible in this scan?",
                lines=2,
            )
            with gr.Row():
                dense_cb = gr.Checkbox(value=True, label="Dense retrieval")
                graph_cb = gr.Checkbox(value=True, label="KG retrieval")
            submit = gr.Button("Ask", variant="primary")

        with gr.Column(scale=1):
            answer_out = gr.Textbox(label="Answer", lines=6)
            with gr.Accordion("Retrieved evidence", open=True):
                evidence_out = gr.Markdown()
            with gr.Accordion("Grounding check", open=True):
                grounding_out = gr.Markdown()

    submit.click(
        answer,
        inputs=[image_in, question_in, dense_cb, graph_cb],
        outputs=[answer_out, evidence_out, grounding_out],
    )

    gr.Markdown(
        "Toggling the retrieval paths off reproduces the ablation rows from the paper — "
        "compare the grounding check with the KG on versus off."
    )

if __name__ == "__main__":
    demo.launch()
