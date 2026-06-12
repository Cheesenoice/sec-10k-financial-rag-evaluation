"""
src/retrieval/hybrid_retriever.py
Tích hợp BM25 Search và Vector Search sử dụng Reciprocal Rank Fusion (RRF)
Kết hợp độ chính xác của từ khóa và độ bao phủ của ngữ nghĩa
"""

from loguru import logger
from src.config import BM25_TOP_K, VECTOR_TOP_K, RRF_K
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_index import VectorIndex

class HybridRetriever:
    def __init__(self, bm25_index: BM25Index, vector_index: VectorIndex):
        self.bm25_index = bm25_index
        self.vector_index = vector_index

    def search(self, query: str, top_k: int = 5, filter_tickers: list = None, filter_years: list = None) -> list:
        """
        Thực hiện tìm kiếm hỗn hợp kết hợp bộ lọc metadata:
        1. BM25 Search -> lấy top-20
        2. Vector Search -> lấy top-20
        3. Reciprocal Rank Fusion (RRF) -> kết hợp và chấm điểm lại
        """
        # Lấy số lượng ứng viên từ config
        k_bm25 = BM25_TOP_K
        k_vector = VECTOR_TOP_K
        
        # 1. Tìm kiếm độc lập trên hai index (truyền bộ lọc xuống)
        bm25_results = self.bm25_index.search(
            query, top_k=k_bm25, filter_tickers=filter_tickers, filter_years=filter_years
        )
        vector_results = self.vector_index.search(
            query, top_k=k_vector, filter_tickers=filter_tickers, filter_years=filter_years
        )
        
        # 2. Áp dụng Reciprocal Rank Fusion (RRF)
        # RRF_Score(d) = sum( 1 / (RRF_K + Rank(d)) )
        rrf_scores = {}
        doc_store = {} # Lưu trữ thông tin chunk để map ngược lại
        
        # Xử lý kết quả BM25
        for rank, res in enumerate(bm25_results):
            doc_id = res["chunk_id"]
            doc_store[doc_id] = res
            
            # Thứ hạng trong mảng lập chỉ mục từ 0 -> Đổi sang 1-based rank (rank + 1)
            rank_score = 1.0 / (RRF_K + (rank + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rank_score
            
        # Xử lý kết quả Vector
        for rank, res in enumerate(vector_results):
            doc_id = res["chunk_id"]
            doc_store[doc_id] = res
            
            rank_score = 1.0 / (RRF_K + (rank + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rank_score
            
        # 3. Sắp xếp các document theo điểm RRF giảm dần
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 4. Trả về danh sách top_k tài liệu sau khi fusion
        hybrid_results = []
        for doc_id, score in sorted_docs[:top_k]:
            original_doc = doc_store[doc_id]
            hybrid_results.append({
                "chunk_id": doc_id,
                "text": original_doc["text"],
                "score": score, # Trả về điểm RRF làm score mới
                "metadata": original_doc["metadata"],
                "retrieval_source": "hybrid"
            })
            
        return hybrid_results
