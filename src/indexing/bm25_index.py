"""
src/indexing/bm25_index.py
Xây dựng chỉ mục BM25 phục vụ tìm kiếm Lexical cải tiến (Ch.3 & Ch.8)
Sử dụng thư viện rank_bm25
"""

import pickle
import nltk
from pathlib import Path
from loguru import logger
from rank_bm25 import BM25Okapi
from src.config import DATA_INDEXES_DIR

# Đảm bảo có dữ liệu token hóa từ của NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

class BM25Index:
    def __init__(self, index_path: Path = DATA_INDEXES_DIR / "bm25_index.pkl"):
        self.index_path = Path(index_path)
        self.bm25 = None
        self.docs = []

    def tokenize(self, text: str) -> list:
        """Tiền xử lý & Token hóa văn bản thô sang list of lowercase tokens"""
        # Sử dụng NLTK word_tokenize để tách từ chuẩn hơn split() thông thường
        tokens = nltk.word_tokenize(text.lower())
        # Loại bỏ các ký tự đặc biệt, chỉ giữ lại từ vựng
        return [t for t in tokens if t.isalnum()]

    def build_index(self, docs: list):
        """Xây dựng BM25 index trên toàn bộ corpus"""
        logger.info(f"Bắt đầu xây dựng BM25 index cho {len(docs)} chunks...")
        self.docs = docs
        
        # Token hóa từng chunk văn bản
        tokenized_corpus = [self.tokenize(doc["text"]) for doc in docs]
        
        # Khởi tạo BM25 Okapi (k1=1.5, b=0.75 mặc định)
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.success("Xây dựng thành công BM25 Index!")

    def save(self):
        """Lưu chỉ mục xuống đĩa"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bm25": self.bm25,
            "docs": self.docs
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Đã lưu BM25 index tại: {self.index_path.name}")

    def load(self):
        """Tải chỉ mục từ đĩa"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file index tại: {self.index_path}")
            
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            
        self.bm25 = data["bm25"]
        self.docs = data["docs"]
        logger.info(f"Đã tải thành công BM25 index. Corpus chứa {len(self.docs)} chunks.")

    def search(self, query: str, top_k: int = 5, filter_tickers: list = None, filter_years: list = None) -> list:
        """Tìm kiếm tài liệu liên quan bằng thang điểm BM25 kết hợp bộ lọc metadata"""
        if self.bm25 is None:
            raise ValueError("Index chưa được khởi tạo.")
            
        tokenized_query = self.tokenize(query)
        
        # Tính điểm BM25 của query đối với tất cả documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sắp xếp toàn bộ index theo điểm giảm dần
        sorted_indices = scores.argsort()[::-1]
        
        results = []
        for idx in sorted_indices:
            score = float(scores[idx])
            if score <= 0.0:
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
                "score": score,
                "metadata": meta
            })
            
            # Dừng khi thu thập đủ top_k ứng viên thỏa mãn bộ lọc
            if len(results) >= top_k:
                break
                
        return results
