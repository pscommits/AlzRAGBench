# Medical Articles based AlzRAGBench — VectorRAG vs. GraphRAG

This is the **Medical Articles arm** of [AlzRAGBench](.), a multi-part project comparing
Retrieval-Augmented Generation (RAG) architectures on an Alzheimer's disease knowledge
corpus. Other team members are running the same comparison on different datasets and
generation approaches (see, e.g., the `Diffusion AlzRAGBench` folder, which repeats this
comparison — plus HybridRAG — using a diffusion language model as the generator). This
folder is my contribution: a head-to-head **VectorRAG vs. GraphRAG** comparison on a
biomedical-article corpus, using a standard autoregressive LLM for generation.

## Why this comparison

VectorRAG and GraphRAG represent two fundamentally different answers to "what should we
retrieve before asking the LLM to answer?":

- **VectorRAG** embeds passages of text and retrieves by similarity. It's simple, fast,
  and strong on single-fact lookup questions where the answer lives inside one
  contiguous passage.
- **GraphRAG** extracts entities and relations into a structured knowledge graph and
  retrieves a relevant subgraph instead of raw text. It's built to handle questions that
  require **connecting facts scattered across multiple documents** — exactly the kind of
  multi-hop reasoning ("how does APOE genotype relate to amyloid clearance and,
  downstream, to tau pathology?") that plain chunk retrieval struggles with.

Running both pipelines over the *same* corpus, the *same* 30-question evaluation set,
and the *same* generation model isolates the retrieval strategy as the only variable —
so any difference in answer quality can be attributed to retrieval, not generation.

## Repository structure

```
Medical_Articles_AlzRAGBench/
├── VectorRAG_MedicalArticles.ipynb   # dense chunk retrieval pipeline + evaluation
├── GraphRAG_MedicalArticles.ipynb    # knowledge graph pipeline + evaluation
└── README.md
```

Each notebook is self-contained: it installs its own dependencies, loads the source
articles and the shared 30-question evaluation set, builds its own retrieval index
(ChromaDB for VectorRAG, Neo4j for GraphRAG), and runs the full evaluation end to end.

**Note on data:** both notebooks expect the source article JSON files (title, abstract,
plus metadata: PMID, journal, year, authors, MeSH terms, subtopic, DOI, URL) and the
30-question evaluation set (each question annotated with `supporting_sources` and, for
graph evaluation, a `reasoning_path`) mounted from Google Drive. These aren't included
in this snapshot — point the paths in Stage 2 of each notebook at your local copies
before running.

## The two pipelines

| | **VectorRAG** | **GraphRAG** |
|---|---|---|
| **Retrieves** | Top-k text chunks by cosine similarity | A local subgraph of entities + relations |
| **Index** | ChromaDB (persistent, on-disk vector store) | Neo4j Aura graph database |
| **Chunking** | 220-word chunks, 40-word overlap | 300-word chunks, 50-word overlap (chunks <40 words dropped) |
| **Embedding model** | `BAAI/bge-base-en-v1.5` — general-purpose passage retriever | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` — biomedical entity embeddings |
| **Entity/relation extraction** | — | Biomedical NER (`d4data/biomedical-ner-all`) + curated MeSH terms → cleaned, SapBERT-deduplicated entities (0.90 cosine threshold); relations extracted by prompting the LLM to emit `HEAD \| RELATION \| TAIL` triples from a fixed vocabulary (`TREATS`, `CAUSES`, `ASSOCIATED_WITH`, `INHIBITS`, ...) |
| **Retrieval strategy** | Embed the question, query ChromaDB for top-k similar chunks | Embed the question, find the most similar graph entities via SapBERT, pull each entity's local subgraph and supporting chunks |
| **Generation model** | Qwen2.5-3B-Instruct (fp16) | Qwen2.5-3B-Instruct (fp16) — same model, so answer-quality differences trace back to retrieval |
| **Best at** | Single-hop facts stated in one contiguous passage | Chaining facts across multiple connected entities/documents |

Both notebooks use an identical prompting discipline: answer strictly from the
retrieved context, stay concise, and say so explicitly if the context doesn't contain
the answer, rather than hallucinate.

## Evaluation methodology

Both pipelines are scored on the same held-out **30-question set** against two axes:

- **Answer quality** — ROUGE-L (lexical/wording overlap with the gold expected answer)
  and BERTScore (embedding-based semantic similarity, more robust to paraphrasing).
- **Retrieval quality** (independent of generation) — Hit@5, Precision@5, Recall@5, and
  MRR, measured against each question's gold supporting evidence: `supporting_sources`
  (source articles) for VectorRAG, `reasoning_path` (graph entities) for GraphRAG. This
  separates *retrieval* failures from *generation* failures.
- **Efficiency** — average end-to-end latency per question.

## Results

### VectorRAG

| Metric | Score | What it measures |
|---|---|---|
| ROUGE-L | **0.226** | Lexical overlap with the expected answer |
| BERTScore | **0.868** | Semantic similarity to the expected answer |
| Hit@5 | **90.0%** | At least one top-5 chunk came from an expected source |
| Precision@5 | **33.3%** | Share of top-5 retrieved chunks that were relevant |
| Recall@5 | **73.3%** | Share of expected sources actually retrieved |
| MRR | **0.844** | How high the first correct source ranked, on average |
| Avg. latency | **8.74 s / question** | |

### GraphRAG

| Metric | Score | What it measures |
|---|---|---|
| ROUGE-L | **0.146** | Lexical overlap with the expected answer |
| BERTScore | **0.848** | Semantic similarity to the expected answer |
| Hit@5 | **50.0%** | At least one top-5 entity matched the reasoning path |
| Precision@5 | **8.4%** | Share of top-5 retrieved entities that were relevant |
| Recall@5 | **14.2%** | Share of expected reasoning-path entities retrieved |
| MRR | **0.262** | How high the first correct entity ranked, on average |
| Avg. latency | **26.7 s / question** | |

### Reading the scorecard

**VectorRAG comes out ahead on every metric in this comparison.** Dense chunk retrieval
found the right supporting articles far more reliably (90% Hit@5 vs. 50%) and ranked
them much higher on average (MRR 0.84 vs. 0.26), which flows straight through to
stronger answers — both in semantic quality (BERTScore 0.868 vs. 0.848) and, more
sharply, in lexical overlap with the gold answer (ROUGE-L 0.226 vs. 0.146). VectorRAG is
also roughly 3x faster (8.7s vs. 26.7s per question), since it skips the
entity-linking → subgraph-fetch → chunk-fetch chain that GraphRAG's retrieval requires.

GraphRAG's semantic answer quality is still solid on its own (BERTScore ~0.85) — the
LLM generates reasonably grounded answers even from sparse graph context. But
**retrieval, not generation, is GraphRAG's bottleneck here**: a Hit@5 of 50% means half
the time entity linking never surfaces a single node on the question's actual reasoning
path, leaving the LLM with no real evidence to answer from (visible in the raw outputs
as explicit "insufficient evidence" responses). The low Precision@5 (8.4%) alongside a
50% hit rate also suggests that even when GraphRAG does find something relevant, it's
usually buried in an otherwise-noisy top-5 list.

This is a useful counterpoint to note against other approaches in the broader
AlzRAGBench project: it isn't a given that GraphRAG beats VectorRAG — the result depends
heavily on how good entity extraction and entity-linking are, and on how well the
underlying knowledge graph is built. A single-embedding-model retrieval strategy
(SapBERT cosine similarity over raw entity strings, no re-ranking, no fuzzy alias
matching) is a fairly basic baseline, so these numbers should be read as a floor for
this specific GraphRAG design rather than a ceiling on what graph-based retrieval can
achieve on this corpus.

## Limitations

- 30 questions is a small eval set; differences of a few points should not be treated as
  statistically robust.
- ROUGE-L and BERTScore are imperfect proxies for factual correctness — they reward
  wording/topical overlap, not verified accuracy.
- GraphRAG's entity extraction combines NER output with MeSH terms and depends on the
  LLM correctly following the `HEAD | RELATION | TAIL` extraction format; malformed or
  missed extractions directly shrink the graph's coverage.
- Entity linking at query time is pure embedding similarity (SapBERT, no fuzzy alias
  matching, no re-ranking) — it will miss entities phrased differently from how they
  appear in the graph.
- This sub-project only compares VectorRAG and GraphRAG; it does not include a
  HybridRAG arm (see the Diffusion AlzRAGBench folder for a full three-way
  VectorRAG/GraphRAG/HybridRAG ablation on a different corpus and generator).

## Setup and running

**VectorRAG notebook** requires: `chromadb`, `sentence-transformers`, `transformers`,
`accelerate`, `bitsandbytes`, `bert-score`, `rouge-score`, `evaluate`, `pandas`,
`numpy`, `matplotlib`, `tqdm`, `scikit-learn`.

**GraphRAG notebook** requires: `transformers`, `accelerate`, `bitsandbytes`,
`sentence-transformers`, `neo4j`, `networkx`, `scikit-learn`, `evaluate`, `bert-score`,
`rouge-score`, `pandas`, `numpy`, `matplotlib`, `tqdm` — plus a running **Neo4j Aura**
instance (or local Neo4j) with connection credentials.

Both notebooks load Qwen2.5-3B-Instruct in fp16, so a GPU with roughly ≥8GB VRAM is
recommended for either.

Suggested run order:

```
jupyter notebook VectorRAG_MedicalArticles.ipynb
jupyter notebook GraphRAG_MedicalArticles.ipynb
```

Each notebook runs its full pipeline — data loading, indexing/graph construction,
retrieval, generation, and the 30-question evaluation — end to end, and exports its
results to CSV (`vectorrag_complete_results.csv`, `graphrag_evaluation_results.csv`) for
downstream comparison.
