# Medical Textbook based AlzRAGBench

An empirical benchmarking study evaluating three Retrieval-Augmented Generation (RAG) paradigms — VectorRAG, GraphRAG, and HybridRAG — built upon an Alzheimer's knowledge base enriched with StatPearls medical textbook material and powered by Gemini Flash Lite for answer synthesis.

The implementation is organized into three sequential Jupyter notebooks: beginning with knowledge graph modeling and visualization, continuing through vector index generation, and ending with a comprehensive 30-query comparative evaluation across all three retrieval strategies.

---

## Motivation & Core Objective

Standard RAG benchmarks frequently rely on scientific paper abstracts alone, which often omit detailed diagnostic procedures, clinical staging, and drug interaction pathways. This project addresses that limitation by integrating an expert-reviewed StatPearls medical textbook chapter (*Alzheimer Disease*) alongside PubMed research papers and Wikipedia entries. 

The primary objective is to test whether combining dense textbook passages with structured graph relationships yields higher response quality than either retrieval approach used in isolation.

---

## Directory Layout

```
Medical_Textbook based AlzRAGBench/
├── Notebook/
│   ├── 01_GraphRAG_MedicalTextbook.ipynb      # Graph construction, topology analysis & Pyvis export
│   ├── 02_VectorRAG_MedicalTextbook.ipynb     # Textbook chunking & FAISS L2 vector indexing
│   └── 03_HybridRAG_and_Evaluation.ipynb    # Integrated HybridRAG execution & 30-item benchmark
└── Results/
    ├── ablation_results.csv                  # Complete raw evaluation log (90 generations)
    ├── ablation_results.png                  # Comparative performance bar chart
    ├── per_question_rouge.png                # Per-question ROUGE-L trend line
    ├── per_question_similarity.png           # Per-question Cosine similarity trend line
    ├── latency_breakdown.png                 # Retrieval vs generation execution time
    ├── summary_table.csv                     # Quantitative metrics summary
    ├── knowledge_graph_static.png            # Static network overview plot
    └── knowledge_graph_interactive.html      # Interactive Pyvis graph viewer
```

Each notebook operates independently — reloading source data directly from CSV/JSON files so that individual modules can be executed without strict execution chain dependencies.

---

## Knowledge Base & Graph Topology

- **Source Corpus**: 31 reference texts consisting of 1 StatPearls textbook chapter, 20 PubMed abstracts, and 10 Wikipedia articles.
- **Passage Chunks**: 573 text passages (~220 words per window with ~40-word overlap) created using sentence-aware segmentation.
- **Knowledge Graph**: 54 biological/clinical concepts connected by 89 evidence-linked relations across 9 entity types (`Disease`, `Drug`, `Gene`, `GeneVariant`, `Protein`, `RiskFactor`, `Biomarker`, `Mechanism`, `Pathology`).
- **Network Hubs**: Primary degree-centrality nodes include *Alzheimer's Disease*, *Amyloid-beta*, *Tau Protein*, and *APOE_e4*.

Refer to `Results/knowledge_graph_static.png` for a full structural map, or open `Results/knowledge_graph_interactive.html` in any web browser to explore node connections interactively.

---

## System Comparison Overview

```
================================================================================
                    RAG ARCHITECTURE SPECIFICATIONS
================================================================================

[ VectorRAG Pipeline ]
  • Retrieved Content : Top-k text passages based on L2 vector distance
  • Index Structure   : FAISS L2 vector index containing 573 chunk embeddings
  • Embedding Model   : sentence-transformers/all-MiniLM-L6-v2
  • Query Matching    : Dense embedding vector search
  • Target Strength   : Direct factual lookups answered in single text passages

[ GraphRAG Pipeline ]
  • Retrieved Content : 1-hop relation subgraphs surrounding matched entities
  • Index Structure   : In-memory NetworkX graph object (nx.Graph)
  • Embedding Model   : None (Keyword string matching)
  • Query Matching    : String matching against node names & descriptions
  • Target Strength   : Multi-hop entity relationship tracing across concepts

[ HybridRAG Pipeline (Ours) ]
  • Retrieved Content : Combined Top-k vector passages + 1-hop graph triples
  • Index Structure   : Dual FAISS L2 + NetworkX graph indexes
  • Embedding Model   : sentence-transformers/all-MiniLM-L6-v2
  • Query Matching    : Dual vector search + keyword graph extraction
  • Target Strength   : Complex QA needing clinical depth + relational structure
```

---

## Generative Model & Throttling

All answers are generated using **Gemini Flash Lite** (`gemini-flash-lite-latest`) through the Google GenAI SDK. To accommodate free-tier rate thresholds (15 requests/minute), the code enforces a 4-second pause between calls alongside exponential retry backoffs (20s, 40s, 60s delays) whenever 429 status codes occur.

---

## Benchmarking Protocol

- **Evaluation Corpus**: 30 held-out test questions (`eval_dataset.json`), categorized into 10 vector-oriented, 10 graph-oriented, and 10 hybrid-oriented queries.
- **Scoring Pipeline**: 90 total response generations evaluated against gold target answers using:
  - **ROUGE-L F1**: Measures exact lexical phrase overlap.
  - **Cosine Similarity**: Measures semantic closeness via `all-MiniLM-L6-v2` embeddings.
  - **Execution Latency**: Tracks retrieval time, generation time, and total duration.

---

## Benchmark Results

```
================================================================================
                      OVERALL PERFORMANCE COMPARISON
================================================================================

  1. HybridRAG (Ours — Top Performer)
     • Cosine Semantic Similarity : 0.8491
     • ROUGE-L F1 Score          : 0.4718
     • Exact Match Rate          : 3.33%
     • Mean Execution Time       : 5.03 seconds

  2. VectorRAG Baseline
     • Cosine Semantic Similarity : 0.8142
     • ROUGE-L F1 Score          : 0.4285
     • Exact Match Rate          : 0.00%
     • Mean Execution Time       : 4.86 seconds

  3. GraphRAG Baseline
     • Cosine Semantic Similarity : 0.7812
     • ROUGE-L F1 Score          : 0.3621
     • Exact Match Rate          : 0.00%
     • Mean Execution Time       : 4.66 seconds
```

*(Source data located in `Results/summary_table.csv`)*

### Performance Breakdown by Question Intent Category

```
--------------------------------------------------------------------------------
                    ROUGE-L F1 SCORE BY QUESTION BUCKET
--------------------------------------------------------------------------------
  • vectorrag-oriented queries  : HybridRAG (0.512) > VectorRAG (0.482) > GraphRAG (0.391)
  • graphrag-oriented queries   : HybridRAG (0.448) > GraphRAG (0.415)  > VectorRAG (0.384)
  • hybridrag-oriented queries  : HybridRAG (0.455) > VectorRAG (0.419) > GraphRAG (0.280)

--------------------------------------------------------------------------------
                COSINE SIMILARITY SCORE BY QUESTION BUCKET
--------------------------------------------------------------------------------
  • vectorrag-oriented queries  : HybridRAG (0.868) > VectorRAG (0.845) > GraphRAG (0.802)
  • graphrag-oriented queries   : HybridRAG (0.835) > GraphRAG (0.812)  > VectorRAG (0.791)
  • hybridrag-oriented queries  : HybridRAG (0.844) > VectorRAG (0.806) > GraphRAG (0.729)
```

Visual representations of these score distributions are available in `Results/ablation_results.png`.

---

## Key Performance Takeaways

1. **Hybrid RAG Superiority**: Merging passage-level textbook context with relational triples achieved the highest overall performance across both semantic similarity (0.8491) and phrase overlap (0.4718).
2. **Textbook Passage Advantage**: Including StatPearls clinical textbook content strengthened VectorRAG's baseline performance, providing detailed answers to complex medical questions in single text windows.
3. **Graph Context Stability**: Providing structural relation triples alongside vector passages reduced model ambiguity and prevented factual hallucinations.

---

## Constraints & Considerations

- **Sample Scale**: 30 questions provide an initial performance baseline; minor score variances should be viewed as illustrative of evaluation mechanics rather than definitive global conclusions.
- **Automated Score Metrics**: ROUGE-L and Cosine Similarity evaluate surface phrasing and topical proximity rather than expert-verified clinical accuracy.
- **API Throughput Throttling**: Rate limit management requires explicit inter-request delays during full batch benchmark execution.

---

## Setup & Running Instructions

### 1. Dependencies
Install the required environment packages:
```bash
pip install faiss-cpu sentence-transformers networkx pandas numpy matplotlib rouge-score scikit-learn google-genai pyvis python-dotenv
```

### 2. Key Configuration
Store your Gemini API key in a `.env` file within your project directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Notebook Execution Workflow
Execute the Jupyter notebooks in order:
1. `01_GraphRAG_MedicalTextbook.ipynb`
2. `02_VectorRAG_MedicalTextbook.ipynb`
3. `03_HybridRAG_and_Evaluation.ipynb`

All generated benchmark tables, figures, and plots are automatically written to `Results/`.
