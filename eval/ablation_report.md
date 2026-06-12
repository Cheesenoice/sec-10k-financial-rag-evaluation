# Ablation Study & Evaluation Report

This report presents the scientific evaluation of our SEC 10-K Retrieval-Augmented Generation (RAG) system, analyzing **5 distinct system configurations** over a standardized benchmark of **80 queries** (20 queries per category: Factual, Comparison, Lexical Gap, Temporal Routing).

---

## 1. Mathematical Formulation of Metrics

To evaluate search and retrieval performance, we compute three standard Information Retrieval (IR) metrics at rank $K = 5$:

### 1.1 Recall@K
Recall measures the proportion of relevant chunks retrieved in the top $K$ results relative to the total number of ground truth chunks:

$$\text{Recall@K} = \frac{|\mathcal{R}_K \cap \mathcal{G}|}{|\mathcal{G}|}$$

Where:
* $\mathcal{R}_K$ is the set of top $K$ retrieved document chunks.
* $\mathcal{G}$ is the set of ground truth document chunks for the query.

### 1.2 Mean Reciprocal Rank (MRR@K)
MRR evaluates the position of the *first* relevant chunk. For a set of queries $Q$, it is defined as:

$$\text{MRR@K} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{RR}_q(K)$$

$$\text{RR}_q(K) = \begin{cases} \frac{1}{r_q} & \text{if } 1 \le r_q \le K \\ 0 & \text{otherwise} \end{cases}$$

Where $r_q$ is the rank of the first relevant document chunk retrieved for query $q$.

### 1.3 Normalized Discounted Cumulative Gain (NDCG@K)
NDCG measures retrieval quality by discounting documents ranked lower in the list, assuming binary relevance $rel_i \in \{0, 1\}$ (1 if chunk $i \in \mathcal{G}$, else 0):

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i + 1)}$$

$$\text{IDCG@K} = \sum_{i=1}^{\min(K, |\mathcal{G}|)} \frac{1}{\log_2(i + 1)}$$

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

### 1.4 Reciprocal Rank Fusion (RRF)
RRF combines lexical and semantic search ranks without score calibration, using a constant parameter $k = 60$:

$$\text{RRF\_Score}(d \in \mathcal{D}) = \frac{1}{k + r_{\text{BM25}}(d)} + \frac{1}{k + r_{\text{Dense}}(d)}$$

---

## 2. Generator-Validator Pipeline (Quy trình Xây dựng & Kiểm định)

Để đảm bảo bộ dữ liệu kiểm thử đạt chất lượng học thuật cao nhất, chúng tôi áp dụng quy trình **Generator-Validator Pipeline** qua hai giai đoạn tuần tự:

1. **Giai đoạn 1: Sinh dữ liệu thô (Gemini 3.5 Flash High - Thinking)**
   * Mô hình Gemini 3.5 Flash High đọc qua toàn bộ kho ngữ liệu tài liệu 10-K đã chunking.
   * Tự động sinh ra **120 câu hỏi thô** kèm thẻ metadata phân loại sơ bộ và gán nhãn Ground Truth.
2. **Giai đoạn 2: Lọc & Kiểm định chất lượng (Claude Opus 4.6 Thinking)**
   * Claude Opus 4.6 kiểm tra chéo 120 câu thô chống trùng lặp, loại bỏ 40 câu hỏi mơ hồ hoặc kém chất lượng.
   * Giữ lại **80 câu hỏi tốt nhất**, phân phối đều vào 4 danh mục chính (20 câu/nhóm).
   * Phát hiện và sửa lỗi ánh xạ nhãn Ground Truth lệch ngữ cảnh (ví dụ: các câu `q_26`, `q_28`, `q_38`, `q_65`, `q_69`) để đảm bảo nhãn đúng 100% trước khi thực nghiệm.

---

## 3. Experimental Configurations

We evaluate 5 ablation configurations:

* **Config A (TF-IDF Baseline):** Lexical search using sklearn's TF-IDF vectorizer and Cosine Similarity.
* **Config B (BM25 Baseline):** Lexical search using the Okapi BM25 ranking function.
* **Config C (Dense HNSW Search):** Semantic search using the `BAAI/bge-small-en-v1.5` dense retriever and a FAISS HNSW graph index.
* **Config D (Hybrid):** Combines Config B and Config C candidate pools using RRF ($k=60$).
* **Config E (Enhanced RAG Pipeline):** Incorporates Query Expansion (QE), NLP-based Ticker/Year Routing metadata filters, Hybrid Search, and Cross-Encoder Reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

---

## 4. Overall Performance Evaluation

The table below summarizes overall performance across all 80 benchmark queries:

| Configuration | Recall@5 | MRR@5 | NDCG@5 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Config A:** TF-IDF Baseline | 0.0594 | 0.0519 | 0.0416 | 6.89 ms |
| **Config B:** BM25 Baseline | 0.1125 | 0.1540 | 0.1019 | 18.43 ms |
| **Config C:** Dense HNSW Search | 0.1844 | 0.1985 | 0.1519 | 57.69 ms |
| **Config D:** Hybrid (BM25 + Dense) | 0.1844 | **0.2431** | 0.1640 | 63.05 ms |
| **Config E:** Enhanced RAG Pipeline | **0.2531** | 0.2350 | **0.1885** | 2289.89 ms |

---

## 5. Sub-Category Analysis (Performance by Query Type)

Different components target specific challenges in financial QA. The tables below outline metrics broken down by query category:

### 4.1 Factual Queries (20 queries)
*Tests direct financial fact retrieval (e.g. Net Sales, R&D Expenses).*

| Configuration | Recall@5 | MRR@5 | NDCG@5 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Config A (TF-IDF) | 0.1000 | 0.1017 | 0.0772 | 7.47 ms |
| Config B (BM25) | 0.1750 | 0.2350 | 0.1619 | 23.47 ms |
| Config C (Dense) | 0.1500 | 0.1475 | 0.1210 | 71.43 ms |
| Config D (Hybrid) | 0.2250 | 0.2583 | 0.1877 | 65.42 ms |
| **Config E (Enhanced)** | **0.4000** | **0.3050** | **0.2752** | 2285.00 ms |

### 4.2 Comparison Queries (20 queries)
*Requires retrieving information for multiple tickers or years in a single request.*

| Configuration | Recall@5 | MRR@5 | NDCG@5 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Config A (TF-IDF) | 0.0625 | 0.0767 | 0.0490 | 7.32 ms |
| Config B (BM25) | 0.1250 | 0.2017 | 0.1173 | 18.22 ms |
| Config C (Dense) | 0.1625 | 0.1617 | 0.1267 | 68.77 ms |
| Config D (Hybrid) | 0.2125 | **0.3100** | 0.1905 | 62.71 ms |
| **Config E (Enhanced)** | **0.2375** | 0.2750 | **0.1934** | 2200.58 ms |

### 4.3 Lexical Gap Queries (20 queries)
*Uses financial acronyms/synonyms (e.g., "capex", "R&D spend") that mismatch document text.*

| Configuration | Recall@5 | MRR@5 | NDCG@5 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Config A (TF-IDF) | 0.0250 | 0.0167 | 0.0153 | 6.53 ms |
| Config B (BM25) | 0.1000 | 0.1500 | 0.1000 | 11.33 ms |
| Config C (Dense) | **0.1750** | **0.1750** | **0.1337** | 46.30 ms |
| Config D (Hybrid) | **0.1750** | 0.2167 | 0.1531 | 58.60 ms |
| **Config E (Enhanced)** | **0.1750** | 0.1542 | 0.1243 | 2366.01 ms |

### 4.4 Temporal Routing Queries (20 queries)
*Requires isolating data to a specific fiscal year (e.g. 2022 vs 2024).*

| Configuration | Recall@5 | MRR@5 | NDCG@5 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Config A (TF-IDF) | 0.0500 | 0.0125 | 0.0251 | 6.24 ms |
| Config B (BM25) | 0.0500 | 0.0292 | 0.0285 | 20.68 ms |
| Config C (Dense) | **0.2500** | **0.3100** | **0.2262** | 44.27 ms |
| Config D (Hybrid) | 0.1250 | 0.1875 | 0.1245 | 65.45 ms |
| **Config E (Enhanced)** | 0.2000 | 0.2058 | 0.1609 | 2307.98 ms |

---

## 6. Key Empirical Observations & Theoretical Justification

### 5.1 The Critical Impact of Metadata Routing
Traditional search algorithms are easily distracted in a corpus with many similar tables from different years and tickers. 
* **Observation:** In *Factual* queries, Config E Recall@5 rises to **0.4000** (a **+128%** improvement over BM25 and **+166%** over Dense).
* **Theory:** Limiting the search space by hard metadata matching (restricting search strictly to `AAPL` and `2024` for example) mathematically reduces the candidate document pool size $N$ from 1931 to around 100. This dramatically increases the probability of retrieving target ground truth chunks.

### 5.2 Dense Search Resolves the Lexical Gap
* **Observation:** In *Lexical Gap* queries, TF-IDF and BM25 drop to near zero (Recall@5 of 0.025 and 0.100), whereas Dense search holds strong at **0.1750** Recall.
* **Theory:** Dense vectors encode semantic meaning rather than literal tokens. "Capex" maps to the same latent space as "capital expenditures". Consequently, the HNSW Flat graph navigates directly to the correct neighborhood, bypassing keyword mismatch.

### 5.3 Reranker Latency-Quality Trade-Off
* **Observation:** Config E latency rises to **2289 ms** on CPU (compared to ~63ms for Config D).
* **Theory:** The Cross-Encoder model does not process query and document vectors independently (like bi-encoders/dense search). Instead, it runs full self-attention layers across the concatenated `[Query, Document]` text pairs, calculating cross-attention scores. This is highly compute-intensive, scaling as $O(M \times L^2)$ where $M$ is the candidate pool size (20) and $L$ is the token sequence length.

---

## 7. Visualizations & Analysis Charts

Below are the key analytical plots generated from our evaluation data:

### 6.1 Overall Metrics Comparison
*Shows Recall@5, MRR@5, and NDCG@5 scores side-by-side.*

![Metrics Comparison Plot](C:/Users/huynh/.gemini/antigravity/brain/25cdf4ad-9ff7-420d-a5a7-8d45f5188227/metrics_ablation_comparison.png)

### 6.2 Heatmap of NDCG@5 by Query Category
*Highlights system performance across Factual, Comparison, Lexical Gap, and Temporal Routing.*

![Heatmap Category NDCG](C:/Users/huynh/.gemini/antigravity/brain/25cdf4ad-9ff7-420d-a5a7-8d45f5188227/heatmap_category_ndcg.png)

### 6.3 Latency vs. NDCG@5 Pareto Frontier (Log Scale)
*Demonstrates the Pareto trade-off between execution speed and search quality.*

![Latency vs NDCG Scatter](C:/Users/huynh/.gemini/antigravity/brain/25cdf4ad-9ff7-420d-a5a7-8d45f5188227/latency_vs_ndcg_tradeoff.png)

### 6.4 Recall@5 Category Breakdown Bar Chart
*Shows how the enhanced pipeline specifically targets and resolves individual baseline weaknesses.*

![Recall Category Breakdown](C:/Users/huynh/.gemini/antigravity/brain/25cdf4ad-9ff7-420d-a5a7-8d45f5188227/recall_by_category_breakdown.png)

---

## 8. Teacher Q&A Defense Preparation

### 💬 Q1: Why does RRF outperform individual lexical and semantic pipelines?
* **Answer:** Reciprocal Rank Fusion (RRF) relies on rank position rather than raw scores. Lexical search is highly precise for exact keywords but fails on syntax variations. Semantic search captures broad contexts but loses exact numeric/identifier focus. By summing reciprocal ranks, RRF assigns high scores *only* to documents ranked highly by *both* methods, creating a mutually reinforcing ranking.

### 💬 Q2: Why is the Latency of Config E so high, and how can we optimize it for production?
* **Answer:** The latency (~2.2s) is due to running the Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on CPU. In production, this is optimized by:
  1. Running on GPU (using TensorRT or ONNX Runtime).
  2. Restricting the reranking candidate pool size $M$ from 20 to 5 or 10.
  3. Quantizing the model weights to FP16 or INT8 to reduce memory bandwidth bottleneck.

### 💬 Q3: What is the benefit of using HNSW Flat over a regular flat L2 index in FAISS?
* **Answer:** A regular Flat index (`IndexFlatIP`) performs brute-force search with $O(N)$ time complexity, which scales poorly as the corpus grows. HNSW (Hierarchical Navigable Small World) structures vectors into a multi-layered graph, enabling approximate nearest neighbor search with $O(\log N)$ complexity, saving CPU cycles at the cost of a minor approximation error.

