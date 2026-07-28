# Reasoning LM based HybridRAG

Alzheimer's-domain HybridRAG built on a purpose-built PubMed knowledge graph and a
matched vector index over the same abstracts. Evaluated on the Alzheimer's slice
of [MIRAGE](https://github.com/Teddy-XiongGZ/MIRAGE).

## What this arm varies

Everything except the generator is held constant against the other approach
folders in this repo: same PubMed corpus, same chunking, same KG schema, same
prompt template, same question set. Only the answering LM differs.

## Pipeline

```
PubMed (10 AD queries)
        │
        ├──► LLM triple extraction ──► entity resolution ──► NetworkX MultiDiGraph
        │                                                          │  01
        └──► chunking ──► PubMedBERT ──► FAISS (cosine)            │
                                             │  03                 │
                                             └────────┬────────────┘
                                                      │
                                    GraphRAG · VectorRAG · HybridRAG (RRF)
                                                      │  04
                                                 evaluation
```

Both retrievers index the **same** abstracts. Without that, a GraphRAG-vs-VectorRAG
comparison would confound retrieval method with corpus, and any difference would
be uninterpretable.

## Notebooks

| Notebook | Does |
|---|---|
| `01_knowledge_graph_construction_and_visualization.ipynb` | Harvests AD abstracts from PubMed, extracts all relations per abstract in one LLM call, merges entities across papers via an alias table plus prefix-blocked fuzzy matching, measures bridge-fact coverage, renders subgraphs |
| `02_graphrag_pipeline_and_retrieval_methods.ipynb` | Entity linking, neighbourhood and bridge retrieval, per-entity chain-of-thought traces |
| `03_vectorrag_pipeline.ipynb` | Chunks the same abstracts, embeds with PubMedBERT, builds and validates the FAISS index |
| `04_hybridrag_and_evaluation.ipynb` | Runs all four arms, fuses with RRF, reports Wilson CIs and McNemar tests |

Run order: 01 → 03 → 02 → 04. Notebook 04 needs the artifacts from 01 and 03.

## Strategies compared

| Key | Evidence supplied |
|---|---|
| `closed_book` | none — parametric knowledge only |
| `graphrag` | KG triples: bridge edges between question and option entities, plus per-entity neighbourhoods, plus CoT traces |
| `vectorrag` | top-*k* FAISS chunks retrieved on stem + options |
| `hybridrag` | both, reciprocal-rank fused |
| `hybridrag_gated` | `hybridrag` evidence passed through a similarity gate — passages and KG facts below a relevance threshold are dropped before fusion, so the model sees a much smaller, higher-precision context (or none at all) instead of a fixed-size budget |

### Why reciprocal rank fusion

A KG edge carries a 1–10 extraction confidence; a chunk carries a cosine
similarity. These are not on a common scale and any normalisation between them
would be arbitrary. RRF discards magnitudes and fuses on rank:

```
RRF(d) = Σ_r  1 / (k + rank_r(d)),   k = 60
```

An item ranked highly by either retriever scores well; one ranked highly by both
wins. The fused list is truncated to `HYBRID_BUDGET`, so the hybrid arm sees
roughly the same *quantity* of evidence as the single-retriever arms — any gain
is better selection, not a longer context. The evidence-volume audit printed in
§12 of notebook 04 confirms this per run.

## Running on HF Jobs

```bash
jupyter nbconvert --to script 04_hybridrag_and_evaluation.ipynb

hf jobs uv run --flavor l4x1 --timeout 6h \
    --secrets HF_TOKEN \
    --env KG_REPO=<your-hf-user>/alzheimers-kg \
    --env GENERATOR=google/medgemma-4b-it \
    --env LIMIT=0 \
    04_hybridrag_and_evaluation.py
```

`l4x1` (24 GB, Ada) holds the 4-bit 27B on a single card and supports bf16.
Prefer it over 2×T4: pre-Ampere cards lack bf16, and sharding a 4-bit model
across two of them triggers fp16 overflow and device-side asserts.

Start with `LIMIT=6` to measure the real per-question rate, then set `LIMIT=0`.

### Configuration

Every knob reads from an environment variable, so no code edits between runs.

| Variable | Default | Purpose |
|---|---|---|
| `STRATEGIES` | `closed_book,graphrag,vectorrag,hybridrag` | Comma-separated; `closed_book` must stay first (McNemar reference) |
| `LIMIT` | `0` | Question cap; 0 = all |
| `SC_SAMPLES` | `1` | >1 enables self-consistency voting |
| `MAX_ENTITIES` | `3` | Entity budget per question |
| `FACTS_PER_ENT` | `8` | Neighbourhood facts per entity |
| `MIN_CONF` | `6` | Drop KG edges below this extraction confidence |
| `VEC_TOPK` | `5` | Chunks retrieved per question |
| `RRF_K` | `60` | RRF constant |
| `HYBRID_BUDGET` | `10` | Evidence items in the fused context |
| `KG_COT` | `1` | Per-entity CoT traces |

## Outputs

Written to `Results/` and pushed to the HF dataset repo:

- `hybridrag_eval_report.pdf` — cover, results table, figures, per-question disagreements
- `hybridrag_eval_results.json` — full config, coverage, stats, per-question records
- `hybridrag_eval_predictions.csv` — one row per question per strategy
- `accuracy_by_strategy.png` — accuracy with 95% Wilson intervals
- `fixed_vs_broke.png` — questions each arm fixed vs broke against the baseline
- `accuracy_by_margin.png` — accuracy bucketed by pre-retrieval model certainty
- `hybrid_fusion_mix.png` — what RRF actually selected into the fused context

## Statistics

**Wilson intervals** rather than normal-approximation: at n≈54 the normal
approximation is unreliable near the tails and can produce bounds outside [0,1].

**McNemar** against the closed-book baseline, because every arm answers the same
questions. Treating two accuracies on identical items as independent samples
overstates the evidence; McNemar looks only at disagreements — how many the
retriever fixed versus broke.

**Fixed/broke reported separately** from the net delta, because they cancel. An
arm that fixes 6 and breaks 6 nets to zero but is not behaving like the baseline,
and that is invisible in the accuracy column alone.

**Closed-book letter margin** is measured once per question from the next-token
distribution over the option letters. It is the diagnostic that explains a null
result: if the mean margin is near 1.0 the model is already certain before seeing
any evidence, and retrieval has no headroom to change the answer regardless of
evidence quality. In that case the fix is harder questions, not better retrieval.

## Interpreting the numbers

At n≈54 a difference needs to be roughly 12 points to clear significance. A 3–5
point gap is suggestive at best, and the confidence intervals will overlap. The
report states this explicitly rather than leading with the headline figure —
which is what makes it defensible under questioning.

## Results

Run metadata: generated **2026-07-24 19:14 UTC**, generator `google/medgemma-4b-it`,
self-consistency **off (greedy)**, runtime **29 min**, n = **54** Alzheimer's MIRAGE
items (MedQA, MedMCQA, BioASQ, MMLU-Med).

This run compared `closed_book` against `hybridrag` and a gated variant
(`hybridrag_gated`) rather than the full four-arm sweep — no standalone
`graphrag`/`vectorrag` numbers are reported here.

| Strategy | Accuracy | 95% CI | Fixed | Broke | p (vs closed-book) |
|---|---|---|---|---|---|
| `closed_book` | 0.6481 | [0.515, 0.762] | — | — | base |
| `hybridrag` | 0.6296 | [0.496, 0.746] | 7 | 8 | 1.0000 |
| `hybridrag_gated` | 0.6481 | [0.515, 0.762] | 1 | 1 | 0.4795 |

Head-to-head, `hybridrag` vs `hybridrag_gated`: fixed 7, broke 6, p = 1.0000.

Best-performing strategy: `closed_book` (tied with `hybridrag_gated`), both at
0.6481 — i.e. neither retrieval arm beats the no-retrieval baseline at this
sample size.

### Retrieval coverage

| Metric | Value |
|---|---|
| KG entity-linked | 52 / 54 |
| KG bridge facts | 14 / 54 |
| Vector mean top-1 cosine | 0.531 |
| Mean closed-book letter margin | 0.9257 |

The mean closed-book margin of 0.9257 is very close to 1.0 — the model is
already near-certain before seeing any retrieved evidence on most questions,
which is consistent with retrieval having little room to move the accuracy
needle in this run (see "Interpreting the numbers" above).

### Evidence volume (mean per question)

| Strategy | Facts | Chunks | Context (chars) |
|---|---|---|---|
| `closed_book` | 0.0 | 0.0 | 0 |
| `hybridrag` | 11.6 | 10.0 | 5941 |
| `hybridrag_gated` | 0.8 | 0.6 | 296 |

The gate is aggressive: it drops the fused context from ~5.9k chars down to
~300 chars on average (a ~20× reduction), which is why `hybridrag_gated`
tracks `closed_book` so closely — most questions end up answered with
little or no retrieved evidence at all.

### Corpus / index size

| Component | Size |
|---|---|
| Knowledge graph | 1039 nodes / 1356 edges |
| Vector index | 1488 chunks (`NeuML/pubmedbert-base-embeddings`) |

### Disagreement patterns

Across the 15 logged disagreements, two clusters stand out:

- **BioASQ drug-effectiveness questions** (Semagacestat, Lanabecestat,
  Verubecestat) account for 6 of the 15 disagreements. `closed_book` answers
  "A" on all of these; gold is split between A and B, and `hybridrag`
  correctly flips several to match gold — this is the clearest case of
  retrieval adding real signal rather than noise.
- **MedQA clinical vignettes** account for most of the rest, where
  `hybridrag` diverges from both `closed_book` and gold, while
  `hybridrag_gated` tends to agree with `closed_book` — consistent with the
  gate suppressing the (apparently unhelpful) extra context that led plain
  `hybridrag` astray on these items.

## Artifacts

Knowledge graph, vector index, and results are published to the HF dataset repo
set by `KG_REPO`.

| File | Produced by |
|---|---|
| `alzheimers_kg.pkl`, `alzheimers_edges.jsonl`, `ad_questions.json` | 01 |
| `ad_faiss.index`, `ad_chunks.jsonl`, `ad_vector_meta.json` | 03 |
| `results/*` | 04 |
