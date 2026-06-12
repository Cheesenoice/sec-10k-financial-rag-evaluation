"""
src/ingestion/chunker.py
Chia nhỏ văn bản từ các section thành các chunk thích hợp với metadata đầy đủ
Sử dụng NLTK sentence tokenizer để không làm đứt đoạn câu giữa chừng
"""

import re
import nltk
from pathlib import Path
from loguru import logger
from src.config import CHUNK_SIZE_DEFAULT, CHUNK_OVERLAP_DEFAULT, CHUNK_SIZE_SMALL

# Download dữ liệu phân đoạn câu của NLTK (chỉ chạy lần đầu)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Đang tải NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
    # NLTK cập nhật từ bản 3.9 có thể cần thêm punkt_tab
    try:
         nltk.download('punkt_tab', quiet=True)
    except:
         pass

class DocumentChunker:
    def __init__(self):
        pass

    def count_tokens(self, text: str) -> int:
        """Đếm số lượng token thô (ước tính bằng từ cách nhau bởi khoảng trắng - chuẩn NLP baseline)"""
        # Tránh dùng tokenizer của model lớn để giữ tốc độ chạy offline nhanh
        return len(re.findall(r"\w+", text))

    def split_into_sentences(self, text: str) -> list:
        """Tách văn bản thành danh sách câu bằng NLTK"""
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_section(self, section_text: str, chunk_size: int, chunk_overlap: int) -> list:
        """
        Chia một section thành các chunks dựa trên câu (Sentence-based chunking).
        Gom các câu lại cho tới khi đạt chunk_size, giữ phần overlap.
        """
        sentences = self.split_into_sentences(section_text)
        chunks = []
        
        current_chunk_sentences = []
        current_chunk_tokens = 0
        
        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)
            # Nếu bản thân câu đó dài hơn chunk_size, chia nhỏ câu theo từ (fallback)
            if sent_tokens > chunk_size:
                words = sentence.split(" ")
                for i in range(0, len(words), chunk_size - chunk_overlap):
                    sub_sentence = " ".join(words[i:i + chunk_size])
                    chunks.append(sub_sentence)
                continue

            if current_chunk_tokens + sent_tokens > chunk_size:
                # Lưu chunk hiện tại
                chunks.append(" ".join(current_chunk_sentences))
                
                # Tạo chunk mới chứa phần overlap từ đuôi của chunk trước
                overlap_sentences = []
                overlap_tokens = 0
                # Lấy ngược các câu cuối cùng cho tới khi chạm mốc overlap mong muốn
                for s in reversed(current_chunk_sentences):
                    s_tokens = self.count_tokens(s)
                    if overlap_tokens + s_tokens <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                        
                current_chunk_sentences = overlap_sentences
                current_chunk_tokens = overlap_tokens
                
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sent_tokens
            
        # Thêm phần chunk còn sót lại ở cuối
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
        return chunks

    def process_document(self, ticker: str, year: int, parsed_sections: dict) -> list:
        """
        Nhận kết quả parse từ 1 document, chia nhỏ từng section và tạo metadata.
        Trả về list of dicts dạng raw chunk.
        """
        processed_chunks = []
        
        for sec_name, text in parsed_sections.items():
            # Quyết định chunk size dựa trên loại Section (Adaptive Chunking)
            # Item 1A (Risk Factors) chứa thông tin rất cô đọng, nên dùng chunk nhỏ
            if sec_name == "Item 1A":
                chunk_size = CHUNK_SIZE_SMALL
                chunk_overlap = 32
            else:
                chunk_size = CHUNK_SIZE_DEFAULT
                chunk_overlap = CHUNK_OVERLAP_DEFAULT
                
            logger.debug(f"Chunking {sec_name} với size={chunk_size}, overlap={chunk_overlap}")
            
            raw_chunks = self.chunk_section(text, chunk_size, chunk_overlap)
            
            for idx, chunk_text in enumerate(raw_chunks):
                token_count = self.count_tokens(chunk_text)
                # Bỏ qua các chunk quá ngắn (chứa ít hơn 10 từ - thường là lỗi format hoặc tiêu đề rác)
                if token_count < 10:
                    continue
                    
                chunk_id = f"{ticker}_{year}_10K_{sec_name.replace(' ', '')}_c{idx:03d}"
                
                # Format chunk chuẩn RAG
                chunk_data = {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "ticker": ticker,
                        "year": int(year),
                        "form_type": "10-K",
                        "section": sec_name,
                        "chunk_index": idx,
                        "token_count": token_count,
                        "source_file": f"{ticker}_{year}_10-K.htm" # Giả định định dạng tải mặc định
                    }
                }
                processed_chunks.append(chunk_data)
                
        logger.info(f"Hoàn tất chunking {ticker} {year}: Tạo ra {len(processed_chunks)} chunks.")
        return processed_chunks
