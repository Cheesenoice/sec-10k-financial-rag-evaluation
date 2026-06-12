"""
scripts/run_evaluation.py
Chương trình đánh giá Ablation Study cho 5 cấu hình RAG sử dụng 80 câu hỏi chuẩn hóa.
Tính toán các chỉ số: Recall@5, MRR@5, NDCG@5 và Latency (ms).
"""
import sys
import os
import json
import time
import re
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Thêm root dir vào path để import src
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Force UTF-8 encoding for standard output/error to prevent UnicodeEncodeError on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


from src.indexing.tfidf_index import TFIDFIndex
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_index import VectorIndex
from src.retrieval.reranker import CrossEncoderReranker
from src.config import DATA_EVAL_DIR, EVAL_RESULTS_DIR, EVAL_FIGURES_DIR, TARGET_YEARS, RRF_K

# Cấu hình log
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# ─── 1. Khởi tạo và nạp các chỉ mục (Chỉ nạp 1 lần) ───
logger.info("Đang khởi tạo các cấu hình chỉ mục tìm kiếm...")
tfidf_index = TFIDFIndex()
tfidf_index.load()

bm25_index = BM25Index()
bm25_index.load()

vector_index = VectorIndex()
vector_index.load()

reranker = CrossEncoderReranker()

# ─── 2. Các hàm bổ trợ logic tìm kiếm nâng cao ───
def expand_query(query: str) -> str:
    """Mở rộng truy vấn giải quyết Lexical Gap"""
    expanded_parts = [query]
    query_lower = query.lower()
    synonym_rules = [
        (r"\b(capital expenditures|capex)\b", "purchases of property and equipment acquisition of property plant and equipment capital spending"),
        (r"\bnet income\b", "net earnings net loss"),
        (r"\brevenue\b", "net sales operating revenue revenues")
    ]
    for pattern, expansion in synonym_rules:
        if re.search(pattern, query_lower):
            expanded_parts.append(expansion)
    return " ".join(expanded_parts)

def route_query(query: str):
    """Phân tích thực thể Ticker & Year từ câu hỏi"""
    detected_tickers = []
    query_lower = query.lower()
    ticker_keywords = {
        "AAPL": ["apple", "aapl"],
        "MSFT": ["microsoft", "msft"],
        "AMZN": ["amazon", "amzn"],
        "NVDA": ["nvidia", "nvda"],
        "TSLA": ["tesla", "tsla"],
        "GOOGL": ["google", "googl", "goog", "alphabet"]
    }
    for ticker, keywords in ticker_keywords.items():
        if any(kw in query_lower for kw in keywords):
            detected_tickers.append(ticker)
            
    year_pattern = r"\b(" + "|".join(map(str, TARGET_YEARS)) + r")\b"
    detected_years = [int(y) for y in re.findall(year_pattern, query)]
    return detected_tickers, detected_years

def rrf(bm25_res, vector_res, top_k=5, rrf_k=60):
    """Kết hợp kết quả bằng Reciprocal Rank Fusion (RRF)"""
    bm25_ranks = {res["chunk_id"]: rank + 1 for rank, res in enumerate(bm25_res)}
    vector_ranks = {res["chunk_id"]: rank + 1 for rank, res in enumerate(vector_res)}
    
    doc_store = {}
    for res in bm25_res:
        doc_store[res["chunk_id"]] = res
    for res in vector_res:
        doc_store[res["chunk_id"]] = res
        
    rrf_scores = {}
    all_doc_ids = set(bm25_ranks.keys()).union(vector_ranks.keys())
    for doc_id in all_doc_ids:
        score = 0.0
        if doc_id in bm25_ranks:
            score += 1.0 / (rrf_k + bm25_ranks[doc_id])
        if doc_id in vector_ranks:
            score += 1.0 / (rrf_k + vector_ranks[doc_id])
        rrf_scores[doc_id] = score
        
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    rrf_candidates = []
    for doc_id, score in sorted_rrf:
        original_doc = doc_store[doc_id]
        rrf_candidates.append({
            "chunk_id": doc_id,
            "text": original_doc["text"],
            "score": score,
            "metadata": original_doc["metadata"],
            "retrieval_source": "hybrid"
        })
    return rrf_candidates[:top_k]

# ─── 3. Định nghĩa 5 cấu hình thử nghiệm (Ablation Configurations) ───
def search_config_a(query: str, top_k=5):
    """Config A: TF-IDF Only"""
    return tfidf_index.search(query, top_k=top_k)

def search_config_b(query: str, top_k=5):
    """Config B: BM25 Only"""
    return bm25_index.search(query, top_k=top_k)

def search_config_c(query: str, top_k=5):
    """Config C: Dense Vector Only"""
    return vector_index.search(query, top_k=top_k)

def search_config_d(query: str, top_k=5):
    """Config D: Hybrid (BM25 + Vector, merged via RRF)"""
    # Lấy top 20 từ mỗi nhánh để hợp nhất qua RRF
    bm25_res = bm25_index.search(query, top_k=20)
    vector_res = vector_index.search(query, top_k=20)
    return rrf(bm25_res, vector_res, top_k=top_k, rrf_k=RRF_K)

def search_config_e(query: str, top_k=5):
    """Config E: Enhanced (QE + Routing + Hybrid + Reranker)"""
    # 1. Tự động routing ticker/year
    tickers, years = route_query(query)
    # 2. Query expansion cho BM25
    expanded_q = expand_query(query)
    
    # 3. Tìm kiếm song song kèm bộ lọc metadata cứng
    bm25_res = bm25_index.search(expanded_q, top_k=20, filter_tickers=tickers, filter_years=years)
    vector_res = vector_index.search(query, top_k=20, filter_tickers=tickers, filter_years=years)
    
    # 4. Hợp nhất RRF
    hybrid_res = rrf(bm25_res, vector_res, top_k=20, rrf_k=RRF_K)
    
    # 5. Rerank bằng Cross-Encoder
    return reranker.rerank(query, hybrid_res, top_k=top_k)

# Mapping configurations
CONFIGS = {
    "Config_A": {"func": search_config_a, "desc": "TF-IDF Baseline"},
    "Config_B": {"func": search_config_b, "desc": "BM25 Baseline"},
    "Config_C": {"func": search_config_c, "desc": "Dense HNSW Search"},
    "Config_D": {"func": search_config_d, "desc": "Hybrid (BM25 + Dense)"},
    "Config_E": {"func": search_config_e, "desc": "Enhanced RAG Pipeline"}
}

# ─── 4. Các chỉ số đánh giá Hệ thống Tìm tin (IR Metrics) ───
def eval_recall_k(retrieved_ids, gt_ids):
    if not gt_ids:
        return 0.0
    intersection = set(retrieved_ids) & set(gt_ids)
    return len(intersection) / len(gt_ids)

def eval_mrr_k(retrieved_ids, gt_ids):
    if not gt_ids:
        return 0.0
    gt_set = set(gt_ids)
    for rank, rid in enumerate(retrieved_ids):
        if rid in gt_set:
            return 1.0 / (rank + 1)
    return 0.0

def eval_ndcg_k(retrieved_ids, gt_ids):
    if not gt_ids:
        return 0.0
    gt_set = set(gt_ids)
    dcg = 0.0
    for rank, rid in enumerate(retrieved_ids):
        if rid in gt_set:
            dcg += 1.0 / math.log2(rank + 2)
            
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(retrieved_ids), len(gt_set))))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

# ─── 5. Hàm chạy kiểm thử 1 câu hỏi ───
def evaluate_query(query_data, config_name, search_func):
    query_id = query_data["query_id"]
    query_text = query_data["query"]
    category = query_data["category"]
    gt_chunks = query_data["ground_truth_chunks"]
    
    start_time = time.perf_counter()
    try:
        results = search_func(query_text, top_k=5)
        latency = (time.perf_counter() - start_time) * 1000  # Đổi sang ms
        retrieved_ids = [res["chunk_id"] for res in results]
        
        recall = eval_recall_k(retrieved_ids, gt_chunks)
        mrr = eval_mrr_k(retrieved_ids, gt_chunks)
        ndcg = eval_ndcg_k(retrieved_ids, gt_chunks)
        
        return {
            "query_id": query_id,
            "category": category,
            "query": query_text,
            "retrieved_ids": retrieved_ids,
            "gt_chunks": gt_chunks,
            "recall": recall,
            "mrr": mrr,
            "ndcg": ndcg,
            "latency_ms": latency
        }
    except Exception as e:
        logger.error(f"Lỗi khi chạy {config_name} cho câu hỏi {query_id}: {e}")
        return None

# ─── 6. Hàm chạy thực nghiệm chính ───
def run_ablation_study():
    # Nạp 80 câu hỏi kiểm thử
    queries = []
    with open(DATA_EVAL_DIR / "test_queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
                
    logger.info(f"Đã nạp {len(queries)} câu hỏi từ test_queries.jsonl")
    
    # Tạo thư mục đầu ra
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    summary_report = {}
    
    for cfg_id, cfg_info in CONFIGS.items():
        logger.info(f"Bắt đầu đánh giá cấu hình: {cfg_id} ({cfg_info['desc']})...")
        search_func = cfg_info["func"]
        
        # Sử dụng ThreadPoolExecutor để chạy song song (an toàn với 4 workers)
        # Giúp tận dụng tốt đa CPU/GPU mà không gây nghẽn phần cứng
        query_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(evaluate_query, q, cfg_id, search_func)
                for q in queries
            ]
            for f in futures:
                res = f.result()
                if res:
                    query_results.append(res)
                    
        # Lưu kết quả thô của cấu hình
        with open(EVAL_RESULTS_DIR / f"{cfg_id}_results.json", "w", encoding="utf-8") as out:
            json.dump(query_results, out, ensure_ascii=False, indent=2)
            
        # Tính toán các chỉ số trung bình tổng thể
        avg_recall = np.mean([r["recall"] for r in query_results])
        avg_mrr = np.mean([r["mrr"] for r in query_results])
        avg_ndcg = np.mean([r["ndcg"] for r in query_results])
        avg_latency = np.mean([r["latency_ms"] for r in query_results])
        
        # Tính toán chỉ số theo từng phân loại câu hỏi (sub-category analysis)
        categories = {}
        for r in query_results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"recalls": [], "mrrs": [], "ndcgs": [], "latencies": []}
            categories[cat]["recalls"].append(r["recall"])
            categories[cat]["mrrs"].append(r["mrr"])
            categories[cat]["ndcgs"].append(r["ndcg"])
            categories[cat]["latencies"].append(r["latency_ms"])
            
        cat_metrics = {}
        for cat, data in categories.items():
            cat_metrics[cat] = {
                "recall@5": round(float(np.mean(data["recalls"])), 4),
                "mrr": round(float(np.mean(data["mrrs"])), 4),
                "ndcg@5": round(float(np.mean(data["ndcgs"])), 4),
                "latency_ms": round(float(np.mean(data["latencies"])), 2)
            }
            
        summary_report[cfg_id] = {
            "desc": cfg_info["desc"],
            "overall": {
                "recall@5": round(float(avg_recall), 4),
                "mrr": round(float(avg_mrr), 4),
                "ndcg@5": round(float(avg_ndcg), 4),
                "latency_ms": round(float(avg_latency), 2)
            },
            "by_category": cat_metrics
        }
        
        logger.success(
            f"Hoàn thành {cfg_id}! "
            f"Recall={avg_recall:.4f} | MRR={avg_mrr:.4f} | NDCG={avg_ndcg:.4f} | Latency={avg_latency:.2f}ms"
        )
        
    # Lưu báo cáo tổng hợp JSON
    with open(EVAL_RESULTS_DIR / "summary_report.json", "w", encoding="utf-8") as out:
        json.dump(summary_report, out, ensure_ascii=False, indent=2)
        
    # ─── 7. Vẽ biểu đồ so sánh (Plotting Charts) ───
    logger.info("Đang vẽ biểu đồ so sánh kết quả thực nghiệm...")
    configs = list(summary_report.keys())
    recalls = [summary_report[c]["overall"]["recall@5"] for c in configs]
    mrrs = [summary_report[c]["overall"]["mrr"] for c in configs]
    ndcgs = [summary_report[c]["overall"]["ndcg@5"] for c in configs]
    latencies = [summary_report[c]["overall"]["latency_ms"] for c in configs]
    
    # Chart 1: So sánh Chất lượng Retrieval (Metrics)
    x = np.arange(len(configs))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, recalls, width, label='Recall@5', color='#4f46e5')
    plt.bar(x, mrrs, width, label='MRR@5', color='#0ea5e9')
    plt.bar(x + width, ndcgs, width, label='NDCG@5', color='#10b981')
    
    plt.ylabel('Chất lượng (Score)')
    plt.title('So sánh các chỉ số tìm kiếm (Retrieval Metrics)')
    plt.xticks(x, [summary_report[c]["desc"] for c in configs])
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(EVAL_FIGURES_DIR / "metrics_comparison.png", dpi=150)
    plt.close()
    
    # Chart 2: Trực quan Latency (Độ trễ)
    plt.figure(figsize=(8, 5))
    plt.bar(x, latencies, color='#f43f5e', width=0.4)
    plt.ylabel('Độ trễ (ms)')
    plt.title('So sánh độ trễ truy vấn (Query Latency)')
    plt.xticks(x, [summary_report[c]["desc"] for c in configs])
    for i, v in enumerate(latencies):
        plt.text(i, v + (max(latencies) * 0.01), f"{v:.1f}ms", ha='center', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(EVAL_FIGURES_DIR / "latency_comparison.png", dpi=150)
    plt.close()
    
    logger.success("Hoàn thành Ablation Study! Các báo cáo và biểu đồ đã được lưu trữ.")

if __name__ == "__main__":
    run_ablation_study()
