# AlzRAGBench

**A Hybrid Knowledge-Graph and Vector Retrieval-Augmented Generation (RAG) Benchmark for Alzheimer's Disease Question Answering.**

AlzRAGBench builds a small, fully-cited Alzheimer's disease corpus from PubMed, Wikipedia, and a medical textbook chapter; turns it into both a **vector index** (VectorRAG) and an LLM-extracted **knowledge graph** (GraphRAG); fuses the two into **HybridRAG** via Reciprocal Rank Fusion; and evaluates the whole pipeline under **four experimental arms** that vary the answering LLM and/or the retrieval corpus. A full write-up with methodology diagrams, exact metrics, and discussion is in [`AlzRAGBench_Summer_Internship_Report.pdf`](./AlzRAGBench_Summer_Internship_Report.pdf) — this README summarises the same project for GitHub.

**Team:** Prabhat Singh, Nathan Pereira, Gurinder Singh, Aishwarya Aryan — under the guidance of Sujoy Kumar Biswas, as a Summer Internship project submitted to IDEAS (Institute of Data Engineering, Analytics and Science Foundation), ISI Kolkata.

## Why

General-purpose LLMs answer medical questions fluently but can hallucinate facts that matter clinically. Retrieval-Augmented Generation grounds answers in retrieved source text instead of relying only on the model's parametric memory. This project asks two questions on the Alzheimer's disease domain: does fusing vector and graph retrieval (HybridRAG) actually beat either alone, and how much does the answer depend on *which* LLM you plug in and *what* you put in the retrieval corpus?

## Architecture

```
PubMed (20) ─┐                                   ┌─► Sentence-aware Chunking (287 chunks)
Wikipedia(10)├─► 31-document corpus ──────────────┤        │
StatPearls(1)┘                                    │        ▼
                                                   │   Dense Embedding → FAISS Index ──► VectorRAG
                    Claude LLM (triple extraction) │
                             │                     │
                             ▼                     │
                  Entity Resolution + Alias Merge  │
                             │                     │
                             ▼                     │
              Knowledge Graph (54 nodes/89 edges) ─┴──► Entity-linked traversal ──► GraphRAG
                                                              │
                                          Reciprocal Rank Fusion (RRF, k=60)
                                                              │
                                                              ▼
                                                HybridRAG Evidence Context
                                                              │
              ┌───────────────┬──────────────────┬───────────┴────────┬──────────────────┐
              ▼               ▼                  ▼                    ▼
        Arm 1: Diffusion  Arm 2: Reasoning   Arm 3: Articles      Arm 4: Textbook
        SDLM-3B-D4        MedGemma-4b-it     Qwen2.5-3B-Instruct  Gemini Flash Lite
        (31-doc corpus)   (MIRAGE, n=54)     (articles only)      (articles+textbook)
              │               │                  │                    │
              └───────────────┴──────────────────┴────────────────────┘
                                        ▼
                     ROUGE-L · BERTScore/CosSim · Hit@5 · MRR · Accuracy+Wilson CI+McNemar
```

The full two-figure architecture diagram (with exact node/edge counts and model names) is in Section 4 of the PDF report.

## Repository structure

```
Dataset/                              Source articles, chunking, knowledge graph, evaluation set,
                                       and the scripts that regenerate all of it from scratch.
Diffusion AlzRAGBench/                Arm 1: diffusion-LM generator (SDLM-3B-D4), notebooks + results.
Reasoning AlzRAGBench/                Arm 2: reasoning-LM generator (MedGemma-4b-it) on MIRAGE, notebooks + results.
Medical_Articles based AlzRAGBench/   Arm 3: VectorRAG vs GraphRAG restricted to article-only chunks.
Medical_Textbook based AlzRAGBench/   Arm 4: VectorRAG/GraphRAG/HybridRAG with textbook chunks included.
```

## Data sources

| Source | Content | Access |
|---|---|---|
| PubMed | 20 abstracts, one per core AD subtopic (pathology, genetics, epidemiology, diagnosis, biomarkers, imaging, MCI, treatment, neuroinflammation, risk factors, synaptic dysfunction, sleep, caregiving, sex differences) | [NCBI E-utilities API](https://www.ncbi.nlm.nih.gov/home/develop/api/) |
| Wikipedia | 10 full articles conceptually overlapping the PubMed subtopics | [Wikipedia Action API](https://www.mediawiki.org/wiki/API:Main_page) |
| StatPearls | "Alzheimer Disease" chapter (public-reuse licence) | [NCBI Bookshelf, NBK499922](https://www.ncbi.nlm.nih.gov/books/NBK499922/) |
| MIRAGE | 54-item Alzheimer's slice (MedQA, MedMCQA, BioASQ, MMLU-Med) used for the Reasoning arm's large-scale evaluation | [MIRAGE benchmark](https://github.com/Teddy-XiongGZ/MIRAGE) |

The knowledge-graph triples and both evaluation question sets were **not** scraped from anywhere — they were generated by prompting **Claude** (Anthropic) over the fetched source text: Claude extracted entity–relation triples for the knowledge graph, and separately authored the 30-question ablation set (gold answers, supporting sources, and KG reasoning paths included).

## Results

All numbers below are recomputed directly from the raw result files in each arm's `Results/` folder.

### Arm 1 — Diffusion LM (SDLM-3B-D4), 31-doc corpus, n=30

| Method | Vector hit-rate | Graph hit-rate | ROUGE-L | Cosine sim. | Time (s) |
|---|---|---|---|---|---|
| VectorRAG | 0.536 | – | 0.228 | 0.712 | 11.92 |
| GraphRAG | – | 0.933 | 0.318 | 0.796 | 8.71 |
| HybridRAG | 0.536 | 0.933 | 0.232 | 0.708 | 13.57 |

**GraphRAG wins outright** — highest answer quality, lowest latency.

### Arm 2 — Reasoning LM (MedGemma-4b-it) on MIRAGE Alzheimer's slice, n=54

| Strategy | Accuracy | 95% CI | Fixed | Broke | p (vs. closed-book) |
|---|---|---|---|---|---|
| closed_book | 0.6481 | [0.515, 0.762] | – | – | baseline |
| hybridrag | 0.6296 | [0.496, 0.746] | 7 | 8 | 1.000 |
| hybridrag_gated | 0.6481 | [0.515, 0.762] | 1 | 1 | 0.480 |

Retrieval does **not** beat the no-retrieval baseline at this scale — the generator is already near-certain before seeing evidence (mean closed-book letter margin 0.9257).

### Arm 3 — Articles-only VectorRAG (Qwen2.5-3B-Instruct), n=30

| Method | Hit@5 | MRR | ROUGE-L | BERTScore | Latency (s) |
|---|---|---|---|---|---|
| VectorRAG | 0.900 | 0.844 | 0.226 | 0.868 | 8.44 |
| GraphRAG | 0.300 | 0.244 | 0.144 | 0.845 | 17.98 |

**VectorRAG dominates** — this GraphRAG uses a weak embedding-only entity linker (no fuzzy matching/re-ranking), so the gap reflects entity-linking quality, not the graph.

### Arm 4 — Textbook-inclusive VectorRAG (Gemini Flash Lite), n=30

| Method | Cosine sim. | ROUGE-L | Exact match | Avg. time (s) |
|---|---|---|---|---|
| VectorRAG | 0.406 | 0.169 | 0.00% | 0.58 |
| GraphRAG | 0.162 | 0.060 | 0.00% | 4.96 |
| HybridRAG | **0.612** | **0.231** | 0.00% | 5.45 |

**HybridRAG wins outright** — +51% cosine similarity over VectorRAG alone once the corpus is diversified with textbook material.

### Cross-arm takeaway

Neither GraphRAG nor HybridRAG is unconditionally better than VectorRAG. GraphRAG's quality is bottlenecked by **entity-linking quality**, not the knowledge graph itself (same 54-node graph wins in Arm 1, loses badly in Arms 3–4 depending only on the linker used). HybridRAG's benefit scales with **generator uncertainty and corpus diversity** — it wins clearly when the corpus is diverse and the generator isn't already confident (Arm 4), and adds nothing measurable when the generator is already near-certain (Arm 2). Full discussion, statistical caveats, and recommended follow-ups are in the [PDF report](./AlzRAGBench_Summer_Internship_Report.pdf), Sections 5–6.

## Reproducing the pipeline

```bash
# 1. Regenerate the corpus, chunks, knowledge graph, and evaluation set
cd Dataset/scripts
python fetch_pubmed.py
python fetch_wikipedia.py
python fetch_textbook.py
python chunk_articles.py
python build_knowledge_graph.py
python build_evaluation_set.py

# 2. Run any of the four experimental arms
# Each arm's notebooks are self-contained; see the arm's own README for run order
# and configuration (e.g. Reasoning AlzRAGBench/README.md documents HF Jobs usage).
```

Each subfolder has its own `README.md` with exact configuration, environment variables, and re-run instructions.

## Citation

If you use this benchmark, please cite the repository:

```
P. Singh, "AlzRAGBench" (GitHub repository), 2026.
https://github.com/pscommits/AlzRAGBench
```

This project builds on the RAG [(Lewis et al., 2020)](https://arxiv.org/abs/2005.11401), GraphRAG [(Edge et al., 2024)](https://arxiv.org/abs/2404.16130), HybridRAG [(Sarmah et al., 2024)](https://arxiv.org/abs/2408.04948), and MIRAGE [(Xiong et al., 2024)](https://github.com/Teddy-XiongGZ/MIRAGE) methodologies, adapted to the Alzheimer's disease domain.

## License

MIT — see [LICENSE](./LICENSE).
