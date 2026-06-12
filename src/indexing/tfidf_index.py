"""
src/indexing/tfidf_index.py
Xây dựng chỉ mục TF-IDF làm baseline truyền thống (Ch.3 & Ch.6)
Sử dụng Cosine Similarity để tìm kiếm độ tương tự
"""

import pickle
from pathlib import Path
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import DATA_INDEXES_DIR

class TFIDFIndex:
    def __init__(self, index_path: Path = DATA_INDEXES_DIR / "tfidf_index.pkl"):
        self.index_path = Path(index_path)
        self.vectorizer = None
        self.tfidf_matrix = None
        self.docs = [] # Lưu trữ danh sách chunks gốc kèm metadata

    def build_index(self, docs: list):
        """
        Xây dựng TF-IDF Vectorizer và tính toán ma trận TF-IDF trên toàn bộ corpus
        """
        logger.info(f"Bắt đầu xây dựng TF-IDF index cho {len(docs)} chunks...")
        self.docs = docs
        
        # Tiền xử lý văn bản: chuyển sang lowercase, loại bỏ English stopwords
        # Stopwords là những từ xuất hiện quá nhiều nhưng mang ít ý nghĩa phân biệt (như: dynamic, the, is, at)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern=r"(?u)\b\w\w+\b" # Token phải chứa từ 2 ký tự trở lên
        )
        
        # Fit và transform corpus để tính ma trận TF-IDF
        # tfidf_matrix sẽ có kích thước: [Số lượng documents, Số lượng từ vựng duy nhất]
        corpus_texts = [doc["text"] for doc in docs]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
        
        logger.success(f"Xây dựng thành công! Từ vựng corpus chứa {len(self.vectorizer.vocabulary_)} từ.")

    def save(self):
        """Lưu chỉ mục xuống đĩa dạng pickle"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "docs": self.docs
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Đã lưu TF-IDF index tại: {self.index_path.name}")

    def load(self):
        """Tải chỉ mục từ đĩa"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file index tại: {self.index_path}")
            
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.docs = data["docs"]
        logger.info(f"Đã tải thành công TF-IDF index. Corpus chứa {len(self.docs)} chunks.")

    def search(self, query: str, top_k: int = 5) -> list:
        """
        Tìm kiếm các chunk liên quan nhất bằng Cosine Similarity
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("Index chưa được xây dựng hoặc chưa tải.")
            
        # 1. Transform query thành TF-IDF vector sử dụng chung từ vựng của Corpus
        query_vector = self.vectorizer.transform([query])
        
        # 2. Tính Cosine Similarity giữa Query vector và toàn bộ ma trận Document vectors
        # cosine_similarity trả về array kích thước: [1, Số lượng documents]
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # 3. Sắp xếp thứ hạng tương đồng giảm dần
        # argsort() trả về index của mảng sắp xếp tăng dần, dùng [::-1] để đảo ngược lại thành giảm dần
        top_indices = similarities.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            # Bỏ qua các tài liệu hoàn toàn không tương thích (score = 0.0)
            if score <= 0.0:
                continue
                
            doc = self.docs[idx]
            results.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "score": score,
                "metadata": doc["metadata"]
            })
            
        return results
