# KG-MMRAG

**Knowledge Graph–Augmented Multimodal Retrieval-Augmented Generation for Medical Imaging**

Research conducted at IDEAS-TIH, Indian Statistical Institute (ISI) Kolkata.

KG-MMRAG grounds medical vision–language generation in two complementary retrieval paths: a dense multimodal index over image–text pairs (BiomedCLIP + FAISS) and a structured clinical knowledge graph (networkx). Retrieved evidence from both paths is fused and passed to MedGemma for generation, with a hallucination scoring module flagging ungrounded claims.

---

## Results

> **TODO(Nathan):** replace every `—` with your actual numbers before sharing. This table is the first thing a reviewer reads.

### Retrieval

| Modality | Dataset | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| Radiology (X-ray) | SLAKE | — | — | — | — |
| MRI (brain) | BraTS | — | — | — | — |
| Multi-modality | ROCOv2 | — | — | — | — |
| _(modality 4)_ | — | — | — | — | — |
| _(modality 5)_ | — | — | — | — | — |
| _(modality 6)_ | — | — | — | — | — |

### Generation & Grounding

| Configuration | Accuracy / F1 | ROUGE-L | BERTScore | Hallucination Rate ↓ |
|---|---|---|---|---|
| MedGemma (no retrieval) | — | — | — | — |
| + Dense retrieval only | — | — | — | — |
| + KG retrieval only | — | — | — | — |
| **+ Fused (KG-MMRAG)** | **—** | **—** | **—** | **—** |

Full metrics: [`results/metrics.json`](results/metrics.json) · Figures: [`results/figures/`](results/figures/)

**Headline finding:** _(TODO: one sentence — e.g. "KG fusion reduces hallucination rate by X points over dense-only retrieval at comparable generation quality.")_

---

## Architecture

```
                    ┌──────────────────┐
   Query (image     │   BiomedCLIP     │
   and/or text) ───▶│    Encoder       │
                    └────────┬─────────┘
                             │ joint embedding
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌────────────────────┐        ┌────────────────────┐
   │  Dense Retrieval   │        │   KG Retrieval     │
   │  FAISS index over  │        │  networkx graph;   │
   │  image–text pairs  │        │  entity linking +  │
   │                    │        │  n-hop traversal   │
   └─────────┬──────────┘        └─────────┬──────────┘
             │  top-k passages              │  subgraph / triples
             └──────────────┬───────────────┘
                            ▼
                 ┌─────────────────────┐
                 │   Evidence Fusion   │
                 │  (rerank + merge)   │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │      MedGemma       │
                 │     Generation      │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Hallucination Score │
                 │ (claim ↔ evidence   │
                 │  entailment)        │
                 └──────────┬──────────┘
                            ▼
                    Grounded answer
                    + evidence trace
```

**Design rationale.** Dense retrieval captures visual and phrasing similarity but is blind to clinical relations — it will happily return a visually similar scan with an unrelated diagnosis. The knowledge graph supplies the relational structure (anatomy → finding → condition) that dense embeddings collapse. Fusing both, then scoring the generated claims against the retrieved evidence, is what lets us report a hallucination rate rather than just a similarity score.

---

## Supported Modalities

| # | Modality | Source dataset | Status |
|---|---|---|---|
| 1 | _(TODO)_ | — | ✅ |
| 2 | _(TODO)_ | — | ✅ |
| 3 | _(TODO)_ | — | ✅ |
| 4 | _(TODO)_ | — | ✅ |
| 5 | _(TODO)_ | — | ✅ |
| 6 | _(TODO)_ | — | ✅ |

---

## Quickstart

```bash
git clone https://github.com/nathanpereira1234/kg-mmrag.git
cd kg-mmrag

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Fetch datasets (see data/README.md for licences and access)
python scripts/prepare_data.py --config configs/default.yaml

# 2. Build the dense index
python scripts/build_index.py --config configs/default.yaml

# 3. Build the knowledge graph
python scripts/build_kg.py --config configs/default.yaml

# 4. Reproduce the results table
python scripts/run_eval.py --config configs/eval.yaml --out results/metrics.json

# 5. Single query (interactive)
python scripts/query.py --image path/to/scan.png --question "What abnormality is visible?"
```

Every number in the Results table is produced by step 4. If it doesn't reproduce, that's a bug — please open an issue.

---

## Repository Layout

```
kg-mmrag/
├── configs/            # All hyperparameters, paths, model IDs. No magic numbers in src/.
│   ├── default.yaml
│   └── eval.yaml
├── data/
│   └── README.md       # How to obtain each dataset. Data itself is gitignored.
├── src/
│   ├── embedding/      # BiomedCLIP wrapper — image & text encoders
│   ├── retrieval/
│   │   ├── dense.py    # FAISS index build + search
│   │   ├── graph.py    # KG construction, entity linking, traversal
│   │   └── fusion.py   # Combines dense + graph evidence
│   ├── generation/     # MedGemma prompting & decoding
│   ├── evaluation/
│   │   ├── metrics.py       # Recall@k, MRR, ROUGE, BERTScore
│   │   └── hallucination.py # Claim-vs-evidence grounding score
│   ├── utils/          # Config loading, logging, seeding
│   └── pipeline.py     # End-to-end orchestration
├── scripts/            # Thin CLI wrappers. The reproducibility contract.
├── results/            # Committed. Metrics + figures reviewers can read without running anything.
├── app/                # Hugging Face Space (Gradio)
├── notebooks/          # Exploratory only — not part of the pipeline
└── tests/
```

---

## Configuration

Nothing is hardcoded. To change a model, dataset path, retrieval depth, or threshold, edit `configs/default.yaml` — not the source.

```yaml
retrieval:
  dense:
    top_k: 10
  graph:
    max_hops: 2
  fusion:
    strategy: weighted   # weighted | rerank | concat
    dense_weight: 0.6
```

---

## Reproducibility

- Seeds fixed in `src/utils/seed.py`; set via `experiment.seed` in config.
- Dependency versions pinned in `requirements.txt`.
- Every run writes its resolved config alongside its metrics, so `results/metrics.json` always records exactly which settings produced it.

---

## Deployment

Live demo: **[Hugging Face Space](https://huggingface.co/spaces/NathanPereira/kg-mmrag)** _(TODO: confirm URL)_

The Space runs the same `src/pipeline.py` as the CLI — `app/app.py` is a thin Gradio layer over it, so the demo and the reported numbers cannot drift apart.

---

## Roadmap

- [ ] _(TODO: current work in progress)_
- [ ] _(TODO: next milestone)_
- [ ] Ablation: KG hop-depth vs. hallucination rate
- [ ] Scale KG to _(TODO)_ entities

---

## Citation

```bibtex
@misc{pereira2026kgmmrag,
  title  = {KG-MMRAG: Knowledge Graph-Augmented Multimodal Retrieval-Augmented
            Generation for Medical Imaging},
  author = {Pereira, Nathan},
  year   = {2026},
  note   = {IDEAS-TIH, Indian Statistical Institute Kolkata},
  url    = {https://github.com/nathanpereira1234/kg-mmrag}
}
```

## Acknowledgements

Work carried out at IDEAS-TIH, ISI Kolkata under the supervision of Dr. Sujoy Kumar Biswas.

## License

MIT — see [LICENSE](LICENSE). Dataset licences are separate and are documented in [`data/README.md`](data/README.md).
