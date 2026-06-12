"""
src/indexing/vector_index.py
Xây dựng chỉ mục Dense Vector Search sử dụng SentenceTransformers và FAISS HNSW (Ch.6)
Tối ưu hóa tìm kiếm láng giềng gần nhất độ phức tạp O(log N)
"""

import pickle
import faiss
import numpy as np
from pathlib import Path
from loguru import logger
from sentence_transformers import SentenceTransformer
from src.config import DATA_INDEXES_DIR, EMBEDDING_MODEL, MODELS_DIR

class VectorIndex:
    def __init__(self, index_dir: Path = DATA_INDEXES_DIR):
        self.index_dir = Path(index_dir)
        self.faiss_index_path = self.index_dir / "vector_index.faiss"
        self.metadata_path = self.index_dir / "vector_metadata.pkl"
        
        # Load embedding model local (tải về models/huggingface/)
        logger.info(f"Đang khởi tạo model Embedding từ {MODELS_DIR}: {EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(MODELS_DIR / "huggingface"))
        self.dimension = self.model.get_sentence_embedding_dimension() # BGE-small là 384 chiều
        
        self.index = None
        self.docs = []

    def build_index(self, docs: list):
        """Xây dựng chỉ mục FAISS HNSW từ danh sách chunks"""
        logger.info(f"Bắt đầu sinh embeddings cho {len(docs)} chunks...")
        self.docs = docs
        
        corpus_texts = [doc["text"] for doc in docs]
        
        # 1. Sinh vector biểu diễn (embeddings) cho toàn bộ corpus
        # show_progress_bar hiển thị thanh tiến trình trực quan
        embeddings = self.model.encode(
            corpus_texts, 
            batch_size=32, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # 2. Chuẩn hóa L2 cho các vector nhúng (đưa độ dài về 1.0)
        # Giúp quy đổi tính toán Cosine Similarity về phép toán nhân ma trận Tích vô hướng (Inner Product) nhanh hơn
        faiss.normalize_L2(embeddings)
        
        # 3. Tạo chỉ mục FAISS HNSW (Hierarchical Navigable Small World)
        # M = 32: Số liên kết tối đa của mỗi node đồ thị láng giềng
        # efConstruction = 200: Số node tối đa khảo sát khi liên kết đồ thị (tăng accuracy khi build)
        # efSearch = 64: Số node khảo sát khi tìm kiếm (tăng accuracy khi search)
        logger.info(f"Đang dựng đồ thị tìm kiếm HNSW (M=32, Dimension={self.dimension})...")
        
        # Sử dụng metric INNER_PRODUCT (tích vô hướng) vì vector đã chuẩn hóa L2
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 64
        
        # Thêm các vector vào đồ thị HNSW
        self.index.add(embeddings.astype("float32"))
        logger.success("Xây dựng thành công Vector HNSW Index!")

    def save(self):
        """Lưu chỉ mục FAISS và metadata đi kèm"""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Lưu file chỉ mục FAISS dạng nhị phân chuyên dụng
        faiss.write_index(self.index, str(self.faiss_index_path))
        
        # Lưu metadata chunks tương ứng để mapping ngược lại sau khi tìm kiếm
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.docs, f)
            
        logger.info(f"Đã lưu Vector Index tại: {self.faiss_index_path.name}")

    def load(self):
        """Tải chỉ mục và metadata từ đĩa"""
        if not self.faiss_index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Không tìm thấy file vector index hoặc metadata.")
            
        self.index = faiss.read_index(str(self.faiss_index_path))
        
        with open(self.metadata_path, "rb") as f:
            self.docs = pickle.load(f)
            
        logger.info(f"Đã tải thành công Vector Index. Đồ thị chứa {self.index.ntotal} nodes.")

    def search(self, query: str, top_k: int = 5, filter_tickers: list = None, filter_years: list = None) -> list:
        """Tìm kiếm láng giềng gần nhất (Approximate Nearest Neighbors - ANN) kết hợp bộ lọc metadata"""
        if self.index is None:
            raise ValueError("Index chưa được khởi tạo.")
            
        # 1. Sinh vector cho query và chuẩn hóa L2
        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        
        # 2. Để tránh bị bộ lọc metadata loại bỏ hết các ứng viên liên quan, 
        # ta quét rộng hơn trên FAISS (ví dụ lấy 200 ứng viên) rồi mới tiến hành lọc.
        search_k = max(top_k * 10, 200)
        # Giới hạn search_k không vượt quá tổng số vector có trong DB
        search_k = min(search_k, self.index.ntotal)
        
        scores, indices = self.index.search(query_vector.astype("float32"), search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
                
            doc = self.docs[idx]
            meta = doc["metadata"]
            
            # Lọc theo Ticker
            if filter_tickers and meta["ticker"] not in filter_tickers:
                continue
            # Lọc theo Năm
            if filter_years and meta["year"] not in filter_years:
                continue
                
            results.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "score": float(score),
                "metadata": meta
            })
            
            # Dừng khi gom đủ top_k ứng viên thỏa mãn bộ lọc
            if len(results) >= top_k:
                break
                
        return results
