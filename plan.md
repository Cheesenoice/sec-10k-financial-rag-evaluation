# RAG Chatbot for SEC 10-K Financial Documents
## Project Implementation Plan & Architecture

---

## 📌 Implementation Overview

| Module / Decision | Chosen Implementation |
|---|---|
| **Domain** | SEC 10-K Filings (Financial Annual Reports) |
| **Data Scale** | 6 companies (AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL) × 3 years (2022, 2023, 2024) |
| **Parsing** | BeautifulSoup4 HTML parsing targeting Items 1, 1A, 7, and 8 |
| **Primary LLM** | Groq Cloud API (Llama 3.3 70B) |
| **Fallback / Local LLM**| Ollama Local (Llama 3.2 3B) |
| **Lexical Search** | BM25 Okapi (`rank_bm25` library) + TF-IDF Cosine Similarity (`scikit-learn` baseline) |
| **Semantic Search** | Dense Retriever (`BAAI/bge-small-en-v1.5`) with FAISS HNSW graph index |
| **Hybrid Search** | Reciprocal Rank Fusion (RRF) with constant parameter $k = 60$ |
| **Metadata Filtering** | Regex & NLP extraction for Ticker & Year Metadata Hard Filters |
| **Reranker** | Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| **Backend API** | FastAPI (running at port 8000) |
| **Frontend UI** | Streamlit chatbot client (running at port 8501) with Live Debug Panel |
| **Evaluation Framework**| 80-query Benchmark Dataset across 4 categories (Factual, Comparison, Lexical Gap, Temporal Routing) |

---

## 🎯 Project Objectives

1. **Academic Rigor:** Strong foundation in Information Retrieval (IR) and Natural Language Processing (NLP) theories (e.g. TF-IDF, BM25, Cosine Similarity, HNSW graphs, Reciprocal Rank Fusion, Cross-Attention scoring).
2. **Quality Benchmark:** Rigorous evaluation through an Ablation Study proving the performance scaling across 5 configurations using standard metrics: Recall@5, MRR@5, and NDCG@5.
3. **Product-Grade UI:** Complete observability of RAG intermediate steps (Lexical/Dense candidate lists, RRF scores, Reranking logits) in a live Streamlit dashboard.

---

## 📦 Data Strategy & Corpus

### 1. Documents Corpus (`data/processed/documents.jsonl`)
* **Total filings:** 18 reports (6 tickers × 3 years).
* **Chunk count:** 1,931 document chunks.
* **Metadata Schema:**
  ```json
  {
    "chunk_id": "AAPL_2024_10K_Item7_c001",
    "text": "...",
    "metadata": {
      "ticker": "AAPL",
      "year": 2024,
      "section": "Item 7",
      "chunk_index": 1
    }
  }
  ```

### 2. Benchmark Queries Dataset (`data/eval/test_queries.jsonl`)
* **Size:** 80 queries, split evenly into 4 categories (20 queries each):
  * **Factual:** Lookup queries for direct financial data (e.g. net sales, R&D expenses).
  * **Comparison:** Cross-company or cross-year comparisons requiring multi-chunk retrieval.
  * **Lexical Gap:** Queries using acronyms/synonyms (e.g., "capex", "R&D spend") that mismatch document vocabulary.
  * **Temporal Routing:** Queries targeting a specific year to evaluate temporal metadata routing.
* **Ground Truth Mappings:** Programmatically mapped and verified to exact, matching `chunk_id` values from the document corpus.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION (Offline)                        │
├────────────────────────────────────────────────────────────────────────┤
│  SEC EDGAR HTML Filings → BeautifulSoup4 → Section Extractor           │
│                                         ↓                              │
│                               Chunking Module                          │
│                                         ↓                              │
│         ┌───────────────────────────────┼────────────────────────┐     │
│         ↓                               ↓                        ↓     │
│    TF-IDF Index                     BM25 Index              FAISS Index│
│   (scikit-learn)                  (rank_bm25)           (bge-small-en) │
└────────────────────────────────────────────────────────────────────────┘
                                          ↓
┌────────────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE (Online)                        │
├────────────────────────────────────────────────────────────────────────┤
│  User Query ──► NLP Ticker/Year Routing (Metadata Hard Filters)        │
│    │                                                                   │
│    ├──► Query Expansion (Lexical Synonyms) ──► BM25 Search (Top-20)    │
│    │                                                                   │
│    └────────────────────────────────────────► Dense Search (Top-20)    │
│                                                     │                  │
│                                                     ▼                  │
│                                          Reciprocal Rank Fusion        │
│                                           (RRF Score, Top-20)          │
│                                                     │                  │
│                                                     ▼                  │
│                                           Cross-Encoder Reranker       │
│                                            (MiniLM Logits, Top-5)      │
│                                                     │                  │
│                                                     ▼                  │
│                                           LLM Generation (Llama 3.3)   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

* **Parser:** BeautifulSoup4 (HTML parsing) & PyMuPDF (PDF fallback).
* **Embeddings:** `bge-small-en-v1.5` via SentenceTransformers (384-dimensional dense vectors).
* **Vector Database:** FAISS (IndexHNSWFlat graph-based approximate nearest neighbors).
* **Keyword Index:** BM25Okapi (`rank_bm25`).
* **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* **LLM Engine:** Groq API (Primary) & Ollama Local Llama 3.2 (Secondary/Evaluation).
* **Application Framework:** FastAPI (Backend) & Streamlit (Frontend).

---

## 📋 Evaluation Results (Ablation Study)

Evaluation metrics computed at rank $K = 5$ over all 80 benchmark queries:

| Configuration | Recall@5 | MRR@5 | NDCG@5 | Avg Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Config A (TF-IDF Baseline)** | 0.0594 | 0.0519 | 0.0416 | 6.89 ms |
| **Config B (BM25 Baseline)** | 0.1125 | 0.1540 | 0.1019 | 18.43 ms |
| **Config C (Dense HNSW)** | 0.1844 | 0.1985 | 0.1519 | 57.69 ms |
| **Config D (Hybrid - RRF)** | 0.1844 | **0.2431** | 0.1640 | 63.05 ms |
| **Config E (Enhanced RAG)** | **0.2531** | 0.2350 | **0.1885** | 2289.89 ms (CPU) |

### Key Theoretical Findings:
1. **BM25 vs. TF-IDF:** BM25 improves Recall@5 by **+89%** due to document length normalization and term frequency saturation.
2. **Dense vs. Lexical:** Dense HNSW overcomes the *Lexical Gap* (e.g. matching "capex" to "capital expenditures"), scoring **0.1750** Recall vs **0.1000** for BM25.
3. **Hybrid Fusion:** RRF (Config D) achieves the highest ranking order (MRR = **0.2431**), combining exact keyword matching with semantic vector space neighbors.
4. **Enhanced RAG:** Config E gains the highest overall Recall (**0.2531**) and NDCG (**0.1885**). NLP Routing restricts the search space (filtering out irrelevant companies/years), eliminating cross-document noise.

---

## 📂 Project Directory Structure

```text
NLP-project/
├── app/
│   └── streamlit_app.py      # Streamlit Frontend UI
├── data/
│   ├── raw/                  # Raw SEC HTML reports
│   ├── processed/            # parsed documents.jsonl
│   ├── indexes/              # Saved index files (BM25, Vector, TF-IDF)
│   └── eval/
│       ├── test_queries.jsonl # 80 benchmark queries
│       └── results/           # Raw JSON outputs of configurations
├── eval/
│   ├── figures/              # Generated ablation charts & plots
│   ├── scripts/
│   │   └── run_evaluation.py # Ablation study evaluation runner
│   └── ablation_report.md    # Academic evaluation report
├── notebooks/
│   └── 4_ablation_study_evaluation.ipynb # Presentation Jupyter Notebook
├── src/
│   ├── api/
│   │   └── main.py           # FastAPI backend
│   ├── indexing/
│   │   ├── bm25_index.py
│   │   ├── tfidf_index.py
│   │   └── vector_index.py
│   ├── retrieval/
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   └── config.py             # Central project configurations
├── requirements.txt          # Python dependencies
├── walkthrough.md            # Interactive walkthrough & Q&A guide
└── plan.md                   # Updated project plan
```

---

## 🚀 Operations & Running Reference

### 1. Start FastAPI Backend API
```powershell
$env:PYTHONUTF8=1
venv\Scripts\python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### 2. Start Streamlit Frontend UI
```powershell
venv\Scripts\streamlit run app/streamlit_app.py
```

### 3. Run Ablation Study
```powershell
venv\Scripts\python eval/scripts/run_evaluation.py
```
