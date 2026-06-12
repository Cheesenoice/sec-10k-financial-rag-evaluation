"""
src/retrieval/reranker.py
Tái xếp hạng (Reranking) sử dụng Cross-Encoder (MiniLM) (Ch.8)
Chạy suy luận sâu so khớp chéo giữa Query và Document
"""

from pathlib import Path
from loguru import logger
from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL, MODELS_DIR

class CrossEncoderReranker:
    def __init__(self):
        # Đường dẫn lưu trữ model cục bộ
        self.model_dir = MODELS_DIR / "huggingface"
        
        logger.info(f"Đang khởi tạo Reranker từ {MODELS_DIR}: {RERANKER_MODEL}...")
        
        # Load model CrossEncoder với cache folder chỉ định
        self.model = CrossEncoder(
            RERANKER_MODEL, 
            cache_folder=str(self.model_dir)
        )

    def rerank(self, query: str, candidates: list, top_k: int = 5) -> list:
        """
        Tái xếp hạng danh sách ứng viên dựa trên điểm tương thích sâu của Cross-Encoder
        """
        if not candidates:
            return []
            
        # 1. Chuẩn bị đầu vào dạng các cặp: [[query, doc1_text], [query, doc2_text], ...]
        pairs = [[query, c["text"]] for c in candidates]
        
        # 2. Dự đoán điểm tương hợp (score càng cao càng liên quan)
        # Điểm số của cross-encoder thường nằm ngoài khoảng [0, 1] (thường là raw logits)
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # 3. Gán điểm mới cho các ứng viên và sắp xếp lại
        reranked_results = []
        for idx, score in enumerate(scores):
            candidate = candidates[idx]
            # Sao chép và cập nhật score cùng nguồn gốc
            updated_doc = {
                "chunk_id": candidate["chunk_id"],
                "text": candidate["text"],
                "score": float(score), # Cập nhật score của Cross-Encoder
                "metadata": candidate["metadata"],
                "retrieval_source": f"reranked({candidate.get('retrieval_source', 'unknown')})"
            }
            reranked_results.append(updated_doc)
            
        # Sắp xếp giảm dần theo điểm score mới
        reranked_results = sorted(reranked_results, key=lambda x: x["score"], reverse=True)
        
        # Trả về top_k kết quả tốt nhất
        return reranked_results[:top_k]
