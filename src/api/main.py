"""
src/api/main.py
FastAPI Backend API cho RAG QA System
Khởi chạy: uvicorn src.api.main:app --reload --port 8000
"""

import sys
import time
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Thêm root dir vào path để import src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.indexing.bm25_index import BM25Index
from src.indexing.vector_index import VectorIndex
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import SYSTEM_PROMPT, USER_TEMPLATE, format_context
from src.config import TARGET_YEARS

app = FastAPI(
    title="SEC 10-K RAG QA System API",
    description="Production-grade Hybrid Retrieval RAG Backend",
    version="1.0.0"
)

# Cấu hình CORS để frontend Streamlit hoặc web app khác có thể gọi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Instances (Sẽ load một lần duy nhất khi startup) ────
bm25_index = None
vector_index = None
hybrid_retriever = None
reranker = None
llm_client = None

@app.on_event("startup")
def startup_event():
    """Load toàn bộ model và chỉ mục khi khởi chạy API"""
    global bm25_index, vector_index, hybrid_retriever, reranker, llm_client
    logger.info("Đang khởi động API backend và nạp dữ liệu chỉ mục...")
    
    try:
        bm25_index = BM25Index()
        bm25_index.load()
        
        vector_index = VectorIndex()
        vector_index.load()
        
        hybrid_retriever = HybridRetriever(bm25_index, vector_index)
        reranker = CrossEncoderReranker()
        
        # Mặc định sử dụng Groq API (Cloud)
        llm_client = LLMClient(use_local=False)
        
        logger.success("API khởi động và nạp chỉ mục thành công!")
    except Exception as e:
        logger.error(f"Khởi động thất bại: {e}")
        # Không crash ứng dụng mà log lỗi để dev kiểm tra
        import traceback
        logger.error(traceback.format_exc())

# ─── Pydantic Models ──────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi cần RAG giải đáp")
    top_k: int = Field(5, description="Số lượng nguồn trích dẫn trả về cuối cùng")
    use_local_llm: bool = Field(False, description="Ép sử dụng Ollama local")
    pipeline_mode: str = Field("enhanced_pipeline", description="Chế độ chạy: baseline_1_lexical, baseline_2_semantic, enhanced_pipeline")
    # Bộ lọc Metadata (Optional)
    filter_tickers: list[str] = Field(default=[], description="Lọc theo mã cổ phiếu (ví dụ: ['AAPL'])")
    filter_years: list[int] = Field(default=[], description="Lọc theo năm tài chính (ví dụ: [2024])")

class Citation(BaseModel):
    chunk_id: str
    text: str
    ticker: str
    year: int
    section: str
    score: float
    source: str = "unknown"

class RRFDetail(BaseModel):
    chunk_id: str
    ticker: str
    year: int
    section: str
    bm25_rank: str
    bm25_score: float
    vector_rank: str
    vector_score: float
    total_score: float

class DebugInfo(BaseModel):
    pipeline_mode: str
    original_query: str
    expanded_query: str | None = None
    detected_tickers: list[str] = []
    detected_years: list[int] = []
    active_keys_count: int = 0
    bm25_raw_results: list[dict] = []
    vector_raw_results: list[dict] = []
    rrf_details: list[RRFDetail] = []
    reranked_results: list[dict] = []

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    llm_source: str
    debug_info: DebugInfo

# ─── Query Expansion ──────────────────────────────────────────
def expand_query(query: str) -> str:
    """
    Mở rộng truy vấn với các thuật ngữ đồng nghĩa tài chính để giải quyết lexical gap.
    """
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

# ─── API Routes ───────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """
    Xử lý câu hỏi qua luồng RAG:
    Có 3 chế độ chạy dựa trên pipeline_mode.
    """
    global bm25_index, vector_index, hybrid_retriever, reranker, llm_client
    
    if bm25_index is None or vector_index is None or hybrid_retriever is None or reranker is None:
        raise HTTPException(status_code=500, detail="Hệ thống chỉ mục chưa được nạp thành công.")
        
    start_total = time.perf_counter()
    
    # Khởi tạo debug info
    debug = DebugInfo(
        pipeline_mode=request.pipeline_mode,
        original_query=request.query
    )
    
    start_retrieve = time.perf_counter()
    
    # ─── CHẾ ĐỘ 1: BASELINE 1 (LEXICAL BM25 ONLY) ───
    if request.pipeline_mode == "baseline_1_lexical":
        logger.info(f"[Mode: Lexical BM25] Query: '{request.query}'")
        raw_candidates = bm25_index.search(
            request.query, top_k=request.top_k, 
            filter_tickers=request.filter_tickers, filter_years=request.filter_years
        )
        final_candidates = []
        for c in raw_candidates:
            c_copy = c.copy()
            c_copy["retrieval_source"] = "bm25"
            final_candidates.append(c_copy)
            
        debug.bm25_raw_results = [
            {"chunk_id": c["chunk_id"], "ticker": c["metadata"]["ticker"], "year": c["metadata"]["year"], "score": c["score"]}
            for c in final_candidates
        ]
        debug.active_keys_count = len(bm25_index.docs)
        
    # ─── CHẾ ĐỘ 2: BASELINE 2 (SEMANTIC HNSW ONLY) ───
    elif request.pipeline_mode == "baseline_2_semantic":
        logger.info(f"[Mode: Semantic Vector] Query: '{request.query}'")
        raw_candidates = vector_index.search(
            request.query, top_k=request.top_k, 
            filter_tickers=request.filter_tickers, filter_years=request.filter_years
        )
        final_candidates = []
        for c in raw_candidates:
            c_copy = c.copy()
            c_copy["retrieval_source"] = "hnsw"
            final_candidates.append(c_copy)
            
        debug.vector_raw_results = [
            {"chunk_id": c["chunk_id"], "ticker": c["metadata"]["ticker"], "year": c["metadata"]["year"], "score": c["score"]}
            for c in final_candidates
        ]
        debug.active_keys_count = len(vector_index.docs)

    # ─── CHẾ ĐỘ 3: ENHANCED PIPELINE (HYBRID + RERANK) ───
    else:
        logger.info(f"[Mode: Enhanced Pipeline] Query: '{request.query}'")
        
        # 1. Query Expansion
        expanded_q = expand_query(request.query)
        debug.expanded_query = expanded_q
        logger.info(f"Expanded Query: '{expanded_q}'")
        
        # 2. NLP Year & Ticker Routing
        detected_tickers = []
        query_lower = request.query.lower()
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
        debug.detected_tickers = detected_tickers
        
        tickers_to_search = detected_tickers if detected_tickers else request.filter_tickers
        
        year_pattern = r"\b(" + "|".join(map(str, TARGET_YEARS)) + r")\b"
        detected_years = [int(y) for y in re.findall(year_pattern, request.query)]
        debug.detected_years = detected_years
        
        years_to_search = detected_years if detected_years else request.filter_years
        
        # 3. Parallel Search
        from src.config import BM25_TOP_K, VECTOR_TOP_K, RRF_K
        bm25_res = bm25_index.search(
            expanded_q, top_k=BM25_TOP_K, filter_tickers=tickers_to_search, filter_years=years_to_search
        )
        vector_res = vector_index.search(
            request.query, top_k=VECTOR_TOP_K, filter_tickers=tickers_to_search, filter_years=years_to_search
        )
        
        debug.bm25_raw_results = [
            {"chunk_id": c["chunk_id"], "ticker": c["metadata"]["ticker"], "year": c["metadata"]["year"], "score": c["score"]}
            for c in bm25_res
        ]
        debug.vector_raw_results = [
            {"chunk_id": c["chunk_id"], "ticker": c["metadata"]["ticker"], "year": c["metadata"]["year"], "score": c["score"]}
            for c in vector_res
        ]
        
        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        doc_store = {}
        bm25_ranks = {res["chunk_id"]: rank + 1 for rank, res in enumerate(bm25_res)}
        vector_ranks = {res["chunk_id"]: rank + 1 for rank, res in enumerate(vector_res)}
        
        for res in bm25_res:
            doc_id = res["chunk_id"]
            doc_store[doc_id] = res
        for res in vector_res:
            doc_id = res["chunk_id"]
            doc_store[doc_id] = res
            
        all_doc_ids = set(bm25_ranks.keys()).union(vector_ranks.keys())
        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in bm25_ranks:
                score += 1.0 / (RRF_K + bm25_ranks[doc_id])
            if doc_id in vector_ranks:
                score += 1.0 / (RRF_K + vector_ranks[doc_id])
            rrf_scores[doc_id] = score
            
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Xây dựng bảng chi tiết RRF để trả về UI
        rrf_details_list = []
        for doc_id, total_score in sorted_rrf:
            doc = doc_store[doc_id]
            bm25_r = bm25_ranks.get(doc_id, -1)
            vector_r = vector_ranks.get(doc_id, -1)
            
            rrf_details_list.append(
                RRFDetail(
                    chunk_id=doc_id,
                    ticker=doc["metadata"]["ticker"],
                    year=doc["metadata"]["year"],
                    section=doc["metadata"]["section"],
                    bm25_rank=str(bm25_r) if bm25_r != -1 else "N/A",
                    bm25_score=1.0 / (RRF_K + bm25_r) if bm25_r != -1 else 0.0,
                    vector_rank=str(vector_r) if vector_r != -1 else "N/A",
                    vector_score=1.0 / (RRF_K + vector_r) if vector_r != -1 else 0.0,
                    total_score=total_score
                )
            )
        debug.rrf_details = rrf_details_list
        debug.active_keys_count = len(all_doc_ids)
        
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
            
        # 4. Cross-Encoder Reranking
        final_candidates = reranker.rerank(request.query, rrf_candidates, top_k=request.top_k)
        
        debug.reranked_results = [
            {"chunk_id": c["chunk_id"], "ticker": c["metadata"]["ticker"], "year": c["metadata"]["year"], "score": c["score"]}
            for c in final_candidates
        ]
        
    retrieval_latency = (time.perf_counter() - start_retrieve) * 1000
    
    # ─── BƯỚC 3: Tạo Prompt & LLM Generation ───
    start_generate = time.perf_counter()
    context_str = format_context(final_candidates)
    user_prompt = USER_TEMPLATE.format(context_str=context_str, query=request.query)
    
    if request.use_local_llm != llm_client.use_local:
         llm_client = LLMClient(use_local=request.use_local_llm)
         
    llm_answer = llm_client.generate(SYSTEM_PROMPT, user_prompt)
    generation_latency = (time.perf_counter() - start_generate) * 1000
    total_latency = (time.perf_counter() - start_total) * 1000
    
    citations = []
    for c in final_candidates:
        citations.append(
            Citation(
                chunk_id=c["chunk_id"],
                text=c["text"],
                ticker=c["metadata"]["ticker"],
                year=c["metadata"]["year"],
                section=c["metadata"]["section"],
                score=c["score"],
                source=c.get("retrieval_source", "unknown")
            )
        )
        
    return QueryResponse(
        answer=llm_answer,
        citations=citations,
        retrieval_latency_ms=round(retrieval_latency, 2),
        generation_latency_ms=round(generation_latency, 2),
        total_latency_ms=round(total_latency, 2),
        llm_source="Ollama (Local Llama 3.2)" if llm_client.use_local else "Groq Cloud (Llama 3.3)",
        debug_info=debug
    )

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}
