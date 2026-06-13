# SEC 10-K RAG QA Chatbot System

**Languages / Ngôn ngữ:** [English (US)](README.md) | [Tiếng Việt (VIE)](README_VIE.md)

---

A high-performance Question-Answering Chatbot system built using Retrieval-Augmented Generation (RAG) to search and analyze SEC 10-K annual financial reports of 6 major technology corporations (AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL) from fiscal years 2022 to 2024.

This repository serves a dual purpose:

1. **Production System:** An end-to-end working system with a FastAPI backend and a Streamlit frontend supporting live debugging and 3 query pipeline routing modes.
2. **Academic Notebooks:** A complete suite of interactive Jupyter Notebooks containing mathematical step-by-step calculations, matrix outputs, and 2D graph visualizations for NLP/IR course presentation.

---

## 📁 Project Directory Structure

```text
NLP-project/
├── data/                     # Data directory
│   ├── raw/                  # Raw SEC EDGAR HTML reports
│   ├── processed/            # parsed documents.jsonl containing clean chunks
│   ├── indexes/              # Saved index files (FAISS .faiss + BM25 .pkl)
│   └── eval/                 # 80 benchmark queries and raw JSON results
├── src/                      # Production RAG backend source code
│   ├── ingestion/            # Data ingestion (downloader, parser, chunker)
│   ├── indexing/             # Search index builders (BM25, Dense HNSW)
│   ├── retrieval/            # Retrieval logic (Hybrid Search, Reranking)
│   ├── generation/           # Prompt templates & LLM API clients
│   └── api/                  # FastAPI web server
├── app/                      # Frontend UI application
│   └── streamlit_app.py      # Streamlit chat interface
├── eval/                     # Evaluation results and reports
│   ├── figures/              # Generated ablation charts & plots
│   ├── scripts/              # Evaluation script runner
│   └── ablation_report.md    # Academic evaluation report
├── notebooks_en/             # Academic Jupyter Notebooks (English)
│   ├── preprocessing/        # Raw parsing and chunking demos
│   ├── baselines/            # Baseline search runs & HNSW graph plots
│   └── 4_ablation_study_evaluation.ipynb # Presentation notebook
├── requirements.txt          # Project dependencies
├── walkthrough.md            # Interactive walkthrough & Q&A guide
└── plan.md                   # Updated project plan
```

---

## 🛠️ Production Source Code Architecture (`src/`)

The backend is modularized into independent RAG pipeline components:

| File Path                           | Core Functionality                                                        | Role in RAG System      |
| :---------------------------------- | :------------------------------------------------------------------------ | :---------------------- |
| `src/ingestion/downloader.py`       | Automatically downloads 10-K filings from the SEC EDGAR API.              | Data Acquisition        |
| `src/ingestion/parser.py`           | Parses SEC HTML files, stripping clutter using BeautifulSoup.             | Preprocessing & Parsing |
| `src/ingestion/chunker.py`          | Splits text into chunks with overlap using sliding window.                | Context Segmentation    |
| `src/indexing/bm25_index.py`        | Builds keyword indexes using the Okapi BM25 ranking algorithm.            | Lexical Indexing        |
| `src/indexing/vector_index.py`      | Generates 384-dim dense vectors (`bge-small-en-v1.5`) & FAISS HNSW graph. | Semantic Indexing       |
| `src/retrieval/hybrid_retriever.py` | Runs parallel keyword/dense queries & combines ranks using RRF.           | Hybrid Retrieval        |
| `src/retrieval/reranker.py`         | Scores cross-attention relevance using a Cross-Encoder.                   | Reranking               |
| `src/generation/llm_client.py`      | Manages connections to Groq Cloud API or Local Ollama.                    | LLM Answer Generation   |
| `src/api/main.py`                   | FastAPI gateway managing NLP routers, expansion, & pipelines.             | API Router Gateway      |

---

## 📓 Academic Jupyter Notebooks (`notebooks/`)

The `notebooks/` folder contains step-by-step walkthroughs using a **sample 5-document corpus** designed to teach the inner workings of NLP/IR algorithms:

### 1. `notebooks/preprocessing/`

- `1_parsing_demo.ipynb`: Regex extraction of Item 1A, Item 7, and Item 8 headers from raw HTML.
- `2_chunking_demo.ipynb`: Visualizing context preservation at boundary edges via sliding windows.

### 2. `notebooks/baselines/`

- `1_lexical/1a_demo_tfidf.ipynb`: Vector Space Model (VSM) calculations:
  - Generates custom Vocabulary dictionary.
  - Prints the complete **TF Matrix**, **IDF Vector**, and **TF-IDF Weight Matrix**.
  - Calculates Cosine Similarity manually and highlights _Lexical Gap_ failures.
- `1_lexical/1b_demo_bm25.ipynb`: Explores the **Okapi BM25** ranking function:
  - Shows how the saturation parameter ($k_1$) and length normalization ($b$) affect ranking.
  - Prints full query-document weight scoring matrices.
- `2_vector/2a_demo_vector.ipynb`: Explores semantic embeddings and vector similarity:
  - Embeds sentences into 384-dimensional spaces using `BAAI/bge-small-en-v1.5`.
  - Prints a $5 \times 5$ document-to-document similarity heatmap.
  - **HNSW 2D Graph Plot:** Performs PCA reduction to 2D and draws the FAISS graph topology using NetworkX.
  - Analyzes the _Temporal Mismatch_ error.
- `3_enhanced/3a_demo_enhanced.ipynb`: Step-by-step run of the full enhanced pipeline:
  - **NLP Year Routing:** Filtering search space via regex.
  - **Query Expansion:** Adding synonyms to lexical queries.
  - **RRF Fusion:** Fraction-level breakdown of Rank Fusion calculations.
  - **Cross-Encoder Reranking:** Showing query-document attention pair logits.

---

## 🚀 Installation & Operation

> [!NOTE]
> **Pre-built Indices Included:** The BM25 index (`bm25_index.pkl`) and FAISS vector index (`vector_index.faiss`) are already built and included in `data/indexes/`. You **do not need to build indices**; simply configure your Groq API key to start querying.

### System Requirements

- Python 3.10+
- Groq API Key (Register for a free key at [Groq Console](https://console.groq.com)).

### 1. Environment Setup

Clone the repository, initialize a virtual environment, and install dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root folder:

```ini
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run Backend API Server (FastAPI)

```powershell
$env:PYTHONUTF8=1
venv\Scripts\python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### 4. Run Frontend Chat Interface (Streamlit)

```powershell
venv\Scripts\streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📈 Evaluation & Ablation Study

Run the ablation runner to verify system configurations (Config A to Config E) against the 80 standard benchmark queries:

```powershell
venv\Scripts\python eval/scripts/run_evaluation.py
```

### Performance Summary

| Configuration                  |  Recall@5  |   MRR@5    |   NDCG@5   |  Latency (Avg)   |
| :----------------------------- | :--------: | :--------: | :--------: | :--------------: |
| **Config A (TF-IDF Baseline)** |   0.2875   |   0.2250   |   0.2258   |     ~6.73 ms     |
| **Config B (BM25 Baseline)**   |   0.4562   |   0.3602   |   0.3512   |     ~24.70 ms    |
| **Config C (Dense HNSW)**      |   0.5000   |   0.4158   |   0.3996   |     ~46.14 ms    |
| **Config D (Hybrid - RRF)**    |   0.5750   |   0.4227   |   0.4467   |     ~76.29 ms    |
| **Config E (Enhanced RAG)**    | **0.7719** | **0.6056** | **0.6184** | ~2002.80 ms (CPU)|

Detailed analysis and visualizations are documented in:

- **Academic Report:** [ablation_report.md](file:///c:/Users/huynh/Desktop/NLP-project/eval/ablation_report.md)
- **Jupyter Evaluation Notebook:** [4_ablation_study_evaluation.ipynb](file:///c:/Users/huynh/Desktop/NLP-project/notebooks_en/eval/4_ablation_study_evaluation.ipynb)
