# Diffusion_AlzRAGBench

**AlzRAGBench** is a controlled ablation study comparing three Retrieval-Augmented
Generation (RAG) architectures — **VectorRAG**, **GraphRAG**, and **HybridRAG** — on a
curated Alzheimer's disease knowledge corpus, using a locally-hosted **diffusion
language model** (not a standard autoregressive LLM) for answer generation.

It is a 4-notebook pipeline that goes from a hand-grounded knowledge graph, through two
independent retrieval pipelines, to a head-to-head evaluation of all three RAG
strategies on a 30-question benchmark designed so that roughly a third of the questions
should favor each method.

## Why this project exists

Most RAG comparisons either (a) only test one retrieval style, or (b) test on a corpus
so large that it's hard to know *why* one method won. This project does the opposite:
a small (31-document), densely cross-referenced, expert-relevant corpus where the
questions are deliberately split into three buckets — `vectorrag`, `graphrag`,
`hybridrag` — each hypothesized in advance to favor a specific retrieval strategy. That
makes it possible to ask a sharper question than "which RAG method is best?": **does
each method actually win on the question type it was designed to win on?**

## Repository structure

```
Diffusion_AlzRAGBench/
├── 01_knowledge_graph_construction_and_visualization.ipynb
├── 02_graphrag_pipeline_and_retrieval_methods.ipynb
├── 03_vectorrag_pipeline.ipynb
├── 04_hybridrag_and_evaluation.ipynb
└── Results/
    ├── raw_results.json              # all 90 generations (30 questions x 3 methods), raw
    ├── ablation_results.csv          # raw_results.json + ROUGE-L / cosine-sim scores
    ├── ablation_results.png          # bar charts: mean score by question bucket x method
    ├── alzheimers_kg.graphml         # exported knowledge graph (Gephi/Cytoscape-readable)
    ├── knowledge_graph_static.png    # matplotlib overview of the full graph
    └── knowledge_graph_interactive.html  # pyvis interactive graph (fully offline HTML)
```

Each notebook is **independently runnable** — every notebook rebuilds the knowledge
graph and/or vector index from the source CSVs/JSON itself rather than depending on a
previous notebook's output, at the cost of a small amount of duplicated setup code.

> **Note on the `Dataset/` folder:** the notebooks expect a sibling `Dataset/` directory
> (`Dataset/Knowledge graph/{nodes.csv,edges.csv}`, `Dataset/chunking/_all_chunks.json`,
> `Dataset/Evaluation/eval_dataset.json`, plus the 31 source documents) that was built in
> an earlier dataset-construction phase of this project. It is not included in this
> snapshot of the repo — add it alongside the notebooks (matching the paths above) before
> running them. The `Results/` folder here contains the already-generated outputs of a
> full run, so the headline results below don't require re-running anything.

## The corpus and knowledge graph

- **31 source documents**: 20 PubMed abstracts, 10 Wikipedia articles, and 1 StatPearls
  textbook chapter on Alzheimer's disease (AD).
- **Knowledge graph**: 54 entities / 89 grounded relations, hand-curated by reading all
  31 documents and citing each edge back to its source article(s). Entity types include
  `Disease`, `Drug`, `Gene`, `GeneVariant`, `Protein`, `RiskFactor`, `Biomarker`,
  `Mechanism`, `Pathology`, `AdverseEffect`, and more (21 types total).
- **Fully connected**: zero isolated nodes, a single connected component — every entity
  is reachable from every other, which is what makes multi-hop GraphRAG traversal
  meaningful rather than a lookup over disconnected fact islands.
- **Hub entities** (highest degree/betweenness centrality): `Alzheimer's Disease`,
  `Amyloid-beta`, and `Tau Protein` — expected, since most of the corpus' concepts
  connect back to these.
- **287 text chunks** for VectorRAG: sentence-boundary-aware ~220-word windows with
  ~40-word overlap over the same 31 documents, each chunk carrying its source
  `article_id` for traceability.

See [`Results/knowledge_graph_static.png`](Results/knowledge_graph_static.png) for a
full overview plot, or open
[`Results/knowledge_graph_interactive.html`](Results/knowledge_graph_interactive.html)
in a browser for a hoverable/draggable version (offline, no server needed).

## The three RAG pipelines

| | VectorRAG (Notebook 3) | GraphRAG (Notebook 2) | HybridRAG (Notebook 4) |
|---|---|---|---|
| **Retrieves** | Top-k text chunks by cosine similarity | A relevant subgraph of entities/relations | Both, independently, merged into one prompt |
| **Index** | In-memory brute-force cosine search (287×768 matrix, no FAISS needed at this scale) | `networkx` `MultiDiGraph` (no Neo4j needed at 54 nodes) | Both indexes above |
| **Embedding model** | `NeuML/pubmedbert-base-embeddings` (biomedical-domain sentence embeddings) | — | `NeuML/pubmedbert-base-embeddings` |
| **Entity linking** | — | Alias-index + word-boundary regex matching (hand-curated abbreviation dictionary, no trained NER) | Same as GraphRAG |
| **Retrieval strategy** | Single strategy: top-k similarity | **Auto-routed** based on how many entities link: 0 → global centrality context, 1 → hub-avoiding multi-hop BFS, 2+ → multi-hop + explicit shortest path | Vector top-k + graph auto-route, concatenated |
| **Best at** | Single-hop facts stated in one place in the text | Chaining facts across multiple connected entities | Questions needing both textual nuance and relational structure |

GraphRAG's retrieval isn't a single fixed algorithm — Notebook 2 implements and
showcases **four distinct strategies** (1-hop neighbor, hub-avoiding multi-hop BFS,
shortest-path, and centrality-guided global context), which Notebook 4 then collapses
into one auto-routing function (`graph_retrieve_auto`) so the ablation doesn't need a
human picking a strategy per question.

### Generation model: SDLM-3B-D4 (a genuine diffusion LLM)

All three pipelines generate answers with **[SDLM-3B-D4](https://huggingface.co/OpenGVLab/SDLM-3B-D4)**
("Sequential Diffusion Language Model", block size 4), a 3B-parameter model fine-tuned
from Qwen2.5-3B — fundamentally different from a standard autoregressive, left-to-right
LLM:

- It's **semi-autoregressive**: it denoises `n_future_tokens` (4, the "D4") masked
  positions at a time in a block, rather than one token at a time.
- Unlike a fully-parallel diffusion model, it keeps a standard causal **KV cache**
  across blocks, reusing previously-computed keys/values — this is what keeps it fast
  and low-memory (runs comfortably in plain bf16, ~6GB, no quantization needed).
- At each step it predicts all masked positions in the block at once, then accepts the
  longest run of *confident* predictions starting from the first; anything past the
  first low-confidence position is discarded and retried next step.

Running it requires a GPU with roughly ≥8GB VRAM. `transformers` is deliberately pinned
to `4.37.2` because SDLM's custom modeling code imports private helpers that later
`transformers` versions removed.

## Evaluation methodology

- **30-question eval set**, evenly split 10/10/10 into questions `designed_to_favor`
  `vectorrag`, `graphrag`, or `hybridrag` respectively (e.g. single-hop factual lookups
  for VectorRAG, multi-hop "trace the chain of events" questions for GraphRAG, and
  broad synthesis questions needing both text nuance and relational structure for
  HybridRAG).
- **90 total generations** (30 questions × 3 methods), each scored two ways:
  - **Answer quality**: ROUGE-L F-measure (wording/phrase overlap with the gold answer)
    and embedding cosine similarity (semantic closeness, using the same PubMedBERT
    model), against an LLM-synthesized (not independently expert-verified) gold
    `expected_answer`.
  - **Retrieval quality** (independent of generation): `vector_hit_rate` — did
    VectorRAG's top-k chunks actually come from the gold `supporting_sources`
    articles? — and `graph_hit_rate` — did GraphRAG's retrieved subgraph actually touch
    the gold `reasoning_path` nodes? This separates *retrieval* failures from
    *generation* failures.

## Results

### Overall mean scores (all 30 questions, per method)

| Method | ROUGE-L | Cosine sim. | Vector hit-rate | Graph hit-rate | Avg. latency |
|---|---|---|---|---|---|
| **GraphRAG** | **0.318** | **0.796** | — | 0.933 | **8.7 s** |
| VectorRAG | 0.228 | 0.712 | 0.536 | — | 11.9 s |
| HybridRAG | 0.232 | 0.708 | 0.536 | 0.933 | 13.6 s |

*(from [`Results/ablation_results.csv`](Results/ablation_results.csv); ROUGE-L and
cosine similarity are computed against the gold `expected_answer`.)*

### Mean score by question bucket × method

**ROUGE-L**

| Question designed to favor... | VectorRAG | GraphRAG | HybridRAG |
|---|---|---|---|
| `vectorrag` | 0.319 | **0.493** | 0.319 |
| `graphrag` | 0.203 | **0.273** | 0.229 |
| `hybridrag` | 0.163 | **0.188** | 0.148 |

**Cosine similarity**

| Question designed to favor... | VectorRAG | GraphRAG | HybridRAG |
|---|---|---|---|
| `vectorrag` | 0.672 | **0.819** | 0.634 |
| `graphrag` | 0.700 | **0.807** | 0.743 |
| `hybridrag` | 0.763 | 0.761 | **0.767** |

See [`Results/ablation_results.png`](Results/ablation_results.png) for the same data as
grouped bar charts.

### Key finding: GraphRAG won almost everywhere — including on VectorRAG's own home turf

The eval set's design hypothesis was only partly confirmed. GraphRAG posted the top
score in 5 of 6 bucket×metric cells above, including beating VectorRAG *on the
questions specifically designed to favor VectorRAG*. Two effects likely explain this:

1. **Retrieval quality was high and roughly symmetric** — mean `graph_hit_rate`
   (0.933) was actually *higher* than mean `vector_hit_rate` (0.536), meaning
   GraphRAG's auto-routed retrieval found its target entities more reliably than
   VectorRAG's top-k search found its target articles.
2. **Prompt length interacted with SDLM's generation quality.** GraphRAG's
   context (a handful of terse `entity --[RELATION]--> entity: evidence` triples) is
   much shorter than VectorRAG's or HybridRAG's (full text passages, or both text +
   graph). Inspecting `raw_results.json` shows VectorRAG and HybridRAG answers
   degenerating into repetition loops (e.g. `"...ant ant ant ant..."`,
   `"...and and and and..."`) noticeably more often than GraphRAG's — consistent with
   SDLM-3B's semi-autoregressive block-diffusion sampling being less robust on longer,
   noisier contexts than on GraphRAG's compact, structured ones. HybridRAG, which
   concatenates *both* context types into the longest prompt of the three, has the
   worst average latency and does not reliably beat plain VectorRAG or GraphRAG.

This is exactly the kind of result the notebook series flags as "not evidence the
dataset or notebook is broken" — it's a genuine finding about how a small diffusion LLM
handles retrieval-augmented context length, which a larger or more robust generator
might not reproduce.

## Limitations (as documented in Notebook 4)

- **30 questions is a small eval set** — mean differences of a few points are not
  statistically robust; treat this as a demonstration of the *evaluation methodology*,
  not a definitive verdict on VectorRAG vs. GraphRAG vs. HybridRAG in general.
- **Automatic metrics are imperfect proxies for correctness.** ROUGE-L rewards wording
  overlap, not factual accuracy; cosine similarity rewards topical closeness, which a
  confidently wrong but related answer can still score well on.
- **Gold answers were LLM-synthesized**, grounded in the source corpus but not
  independently expert-verified.
- **Entity linking is alias/regex-based, not trained NER** — it will miss entities
  mentioned only implicitly or via unanticipated phrasing.
- **HybridRAG's fusion is the simplest possible strategy** (concatenate both contexts,
  let the LLM decide) — no learned re-ranking or weighting between the two evidence
  sources, and no tracking of which half a HybridRAG answer actually drew on.
- **`max_gen_len` was reduced** for the full 90-generation run to keep runtime
  tractable, trading some answer quality (across all methods) for speed.

## Setup and running

**Requirements:**
- Python with `torch`, `transformers==4.37.2` (pinned — see above), `accelerate`,
  `sentence-transformers`, `pandas`, `networkx`, `numpy`, `rouge-score`, `pyvis`,
  `matplotlib`, `tqdm`
- A GPU with ≈8GB+ VRAM for any notebook section that loads SDLM-3B-D4 (Notebook 1 is
  the only fully CPU-only notebook)
- The sibling `Dataset/` directory described above

**Suggested run order:**

```bash
jupyter notebook 01_knowledge_graph_construction_and_visualization.ipynb
jupyter notebook 02_graphrag_pipeline_and_retrieval_methods.ipynb
jupyter notebook 03_vectorrag_pipeline.ipynb
jupyter notebook 04_hybridrag_and_evaluation.ipynb
```

Notebook 4's full ablation run performs 90 SDLM generations and reloads the model every
15 generations as a VRAM safety net; a 3-question "quick test" mode (Section 4.8) lets
you sanity-check the pipeline first. Results are saved incrementally to
`artifacts/raw_results.json` so an interrupted run can resume.
