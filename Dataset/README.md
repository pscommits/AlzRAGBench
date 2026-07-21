# Alzheimer's HybridRAG Dataset

Dataset for building and ablation-testing VectorRAG vs GraphRAG vs HybridRAG on an
Alzheimer's disease corpus.

## Folder structure

```
Selected articles/    31 raw source documents (20 PubMed + 10 Wikipedia + 1 textbook)
chunking/             287 word-window chunks for vector embedding, one file per article + one flat file
Knowledge graph/       54-node / 89-edge entity-relation graph (Neo4j-import-ready CSVs)
Evaluation/            30-item QA ablation dataset (VectorRAG vs GraphRAG vs HybridRAG)
scripts/               Python scripts that generated everything above (rerunnable)
```

## Sources (`Selected articles/`)

- **20 PubMed abstracts** (`pubmed_<pmid>_<subtopic>.json`/`.txt`) — fetched live via NCBI
  E-utilities, one per core Alzheimer's subtopic (pathology, genetics, epidemiology,
  diagnosis, biomarkers, imaging, MCI, treatment classes, neuroinflammation, risk
  factors, synaptic dysfunction, sleep, caregiving, sex differences). Abstract + full
  metadata (authors, journal, year, MeSH terms, DOI) — PubMed only exposes full text
  for the open-access PMC subset, so abstracts were used for consistent coverage.
- **10 Wikipedia articles** (`wikipedia_<slug>.json`/`.txt`) — full article text via the
  Wikipedia Action API, chosen to conceptually overlap with the PubMed subtopics
  (Alzheimer's disease, amyloid beta, tau protein, APOE, dementia, MCI, cholinesterase
  inhibitors, memantine, neurofibrillary tangles, lecanemab).
- **1 textbook chapter** (`textbook_statpearls_alzheimer_disease.json`/`.txt`) — the
  StatPearls "Alzheimer Disease" chapter (NCBI Bookshelf, NBK499922), used as the
  second VectorRAG source. StatPearls content is freely reusable (public-reuse license
  via StatPearls Publishing / NCBI Bookshelf), unlike most commercial textbooks.
- `_pubmed_manifest.json` / `_wikipedia_manifest.json` — index of what was fetched.

Regenerate with: `python scripts/fetch_pubmed.py`, `fetch_wikipedia.py`, `fetch_textbook.py`.

## Chunking (`chunking/`)

287 chunks total across all 31 sources. Strategy: sentence-boundary-respecting word
windows, **~220 words per chunk with ~40-word overlap** (approximates ~300-token chunks
for typical embedding models). Each chunk file (`<article_id>__chunks.json`) contains
chunk text plus full source metadata (title, source_type, authors/URL, etc.) so
retrieval results carry citation info. `_all_chunks.json` is a flat file with every
chunk — convenient for building one vector index across the whole corpus.
`_chunking_manifest.json` records the chunking parameters and per-article chunk counts.

Regenerate with: `python scripts/chunk_articles.py`.

## Knowledge graph (`Knowledge graph/`)

`nodes.csv` (54 rows) and `edges.csv` (89 rows), formatted for `neo4j-admin import` /
`LOAD CSV`:

- `nodes.csv`: `nodeId:ID, name, type:LABEL, description`
- `edges.csv`: `:START_ID, :END_ID, :TYPE, source_articles, evidence`

Every edge cites the `article_id`(s) it's grounded in (`source_articles`, semicolon-
separated) plus a short paraphrased `evidence` string — built by reading the actual
fetched source text, not automated NER, so relations are accurate rather than
statistically inferred. All 31 source articles are cited by at least one edge, and the
graph has **zero isolated nodes** — every entity connects back to the core `AD` node
through at least one path, which is what makes it usable for multi-hop GraphRAG
retrieval rather than a set of disconnected fact islands.

Node types span diseases, proteins/genes, pathology, brain regions, cell types,
mechanisms/theories, drugs/drug classes, biomarkers, risk/protective factors, and
social/economic impact — load directly into Neo4j or any graph library that accepts
edge lists (networkx, igraph, etc.).

Regenerate with: `python scripts/build_knowledge_graph.py` (includes a connectivity +
source-coverage check on every run).

## Evaluation (`Evaluation/eval_dataset.json`)

30 QA pairs for the VectorRAG vs GraphRAG vs HybridRAG ablation study, split evenly:

| Split | Count | Question type | Hypothesis |
|---|---|---|---|
| `vectorrag` | 10 | Single-hop fact lookup, answerable from one chunk | Should favor pure vector retrieval |
| `graphrag` | 10 | Multi-hop/relational, connects 2+ entities not co-located in one chunk | Should favor graph traversal |
| `hybridrag` | 10 | Broad synthesis across many sources + relational reasoning | Should need both |

Each question has:
- `expected_answer` — a synthesized gold reference (paraphrased across sources, not a
  verbatim quote), ready to use directly for scoring.
- `supporting_sources` — `article_id` values matching files in `Selected articles/` and
  `chunking/`, for scoring VectorRAG's retrieved chunks against ground truth.
- `reasoning_path` — `Knowledge graph/nodes.csv` node IDs relevant to the question, for
  scoring GraphRAG's retrieved subgraph against ground truth.
- `difficulty` and `designed_to_favor` — the retrieval method a question is hypothesized
  to favor is **not a guarantee**; use it to check whether your actual ablation results
  match expectations, and treat surprises (e.g. HybridRAG losing on a `hybridrag`
  question) as interesting findings, not dataset bugs.

Both `reasoning_path` and `supporting_sources` references were validated to exist in
`Knowledge graph/nodes.csv` and `Selected articles/` respectively.

Regenerate with: `python scripts/build_evaluation_set.py`.

## Suggested ablation setup

1. **VectorRAG**: embed `chunking/_all_chunks.json` (or per-source-type subsets to test
   PubMed-only / Wikipedia-only / textbook-only VectorRAG) into a vector store.
2. **GraphRAG**: load `Knowledge graph/nodes.csv` + `edges.csv` into Neo4j or networkx;
   retrieve via entity linking + subgraph traversal from the question's mentioned
   entities.
3. **HybridRAG**: combine both — e.g. vector retrieval for supporting text plus graph
   traversal for relational context, or graph-guided re-ranking of vector hits.
4. Run all three against `Evaluation/eval_dataset.json`, score against
   `expected_answer` (e.g. via LLM-judge or ROUGE/BERTScore), and compare performance
   by `designed_to_favor` bucket to see whether each method wins on the question type
   it was expected to.

## Regenerating everything from scratch

```
python scripts/fetch_pubmed.py
python scripts/fetch_wikipedia.py
python scripts/fetch_textbook.py
python scripts/chunk_articles.py
python scripts/build_knowledge_graph.py
python scripts/build_evaluation_set.py
```

Requires `requests` and `beautifulsoup4` (both already installed in this environment).
