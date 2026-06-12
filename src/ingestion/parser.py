"""
src/ingestion/parser.py
Đọc file HTML 10-K và tách thành các section lớn: Item 1, Item 1A, Item 7, Item 8
"""

import re
from pathlib import Path
from bs4 import BeautifulSoup, Tag
from loguru import logger

class SECParser:
    def __init__(self):
        # Regex để tìm ranh giới các Section
        # Sử dụng re.IGNORECASE và chú ý loại bỏ Table of Contents (TOC)
        self.sections_regex = {
            "Item 1": re.compile(r"^\s*Item\s*1[.\s:-]+Business", re.IGNORECASE),
            "Item 1A": re.compile(r"^\s*Item\s*1A[.\s:-]+Risk\s*Factors", re.IGNORECASE),
            "Item 7": re.compile(r"^\s*Item\s*7[.\s:-]+Management[\u2019\'s]*\s*Discussion\s*and\s*Analysis", re.IGNORECASE),
            "Item 8": re.compile(r"^\s*Item\s*8[.\s:-]+Financial\s*Statements", re.IGNORECASE),
            # Section tiếp theo để làm điểm dừng (End anchor)
            "Item 9": re.compile(r"^\s*Item\s*9[.\s:-]+Changes", re.IGNORECASE)
        }

    def clean_html(self, html_content: str) -> str:
        """Lọc bỏ các thẻ script, style và convert HTML entity"""
        soup = BeautifulSoup(html_content, "lxml")
        
        # Loại bỏ các tags rác không cần thiết cho RAG
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
            
        # Trích xuất text có cấu trúc xuống dòng tốt hơn get_text() thông thường
        # Thay thế các thẻ block-level bằng newline để giữ khoảng cách văn bản
        for block_tag in ["p", "div", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "li"]:
            for element in soup.find_all(block_tag):
                element.append("\n")
                
        # Loại bỏ các khoảng trắng thừa
        text = soup.get_text()
        clean_lines = []
        for line in text.split("\n"):
            line = line.strip()
            # Thay thế non-breaking spaces và spaces thừa
            line = re.sub(r"\s+", " ", line)
            if line:
                clean_lines.append(line)
                
        return "\n".join(clean_lines)

    def extract_sections(self, file_path: Path) -> dict:
        """
        Đọc file HTML, parse text và trích xuất các section chính.
        Trả về dict: { "Item 1": "text...", "Item 1A": "text..." }
        """
        logger.info(f"Đang parse file: {file_path.name}")
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
            
        # 1. Clean HTML sang plain text có giữ dòng
        full_text = self.clean_html(html_content)
        lines = full_text.split("\n")
        
        # 2. Tìm dòng chứa tiêu đề các mục (bỏ qua Table of Contents)
        # Table of Contents thường nằm ở 15% phần đầu tài liệu
        min_line_idx = int(len(lines) * 0.10)
        
        section_starts = {}
        
        for idx, line in enumerate(lines):
            # Chỉ bắt đầu quét khi đã đi qua Table of Contents
            if idx < min_line_idx:
                continue
                
            for sec_name, regex in self.sections_regex.items():
                if regex.match(line):
                    # Tránh ghi đè nếu bắt gặp tiêu đề trùng lặp sau đó
                    if sec_name not in section_starts:
                        section_starts[sec_name] = idx
                        logger.debug(f"Tìm thấy start của {sec_name} tại dòng {idx}: '{line[:50]}'")

        # 3. Tách nội dung giữa các ranh giới dòng đã tìm được
        extracted_data = {}
        sec_names = ["Item 1", "Item 1A", "Item 7", "Item 8"]
        
        # Sắp xếp các mốc xuất hiện theo thứ tự dòng tăng dần
        sorted_starts = sorted(section_starts.items(), key=lambda x: x[1])
        
        for i, (sec_name, start_idx) in enumerate(sorted_starts):
            if sec_name not in sec_names:
                continue # Bỏ qua Item 9 vì chỉ dùng làm điểm dừng
                
            # Điểm dừng là vị trí của section tiếp theo trong danh sách sorted
            end_idx = len(lines)
            if i + 1 < len(sorted_starts):
                end_idx = sorted_starts[i+1][1]
                
            section_content = "\n".join(lines[start_idx:end_idx])
            extracted_data[sec_name] = section_content
            logger.info(f"Đã trích xuất {sec_name}: {len(section_content)} ký tự")
            
        # --- SMART FALLBACK CHO NVIDIA / TESLA ---
        # Nếu Item 8 quá ngắn (thường < 2000 ký tự) và có tìm thấy Item 9/Item 15 ở sau
        # Ta quét thêm toàn bộ nội dung từ Item 15 (thường chứa Consolidated Financial Statements ở Part IV) để gộp vào Item 8
        if "Item 8" in extracted_data and len(extracted_data["Item 8"]) < 2000:
            logger.warning(f"Phát hiện Item 8 của {file_path.name} quá ngắn ({len(extracted_data['Item 8'])} ký tự). Tiến hành tìm kiếm Part IV / Item 15...")
            
            # Khởi tạo Regex cho Item 15 (Part IV)
            item15_regex = re.compile(r"^\s*Item\s*15[.\s:-]+Exhibit", re.IGNORECASE)
            item15_start_idx = None
            
            for idx, line in enumerate(lines):
                if idx < min_line_idx:
                    continue
                if item15_regex.match(line):
                    item15_start_idx = idx
                    break
                    
            if item15_start_idx:
                # Gộp toàn bộ nội dung từ Item 15 đến cuối file vào Item 8
                financials_part4 = "\n".join(lines[item15_start_idx:])
                extracted_data["Item 8"] = extracted_data["Item 8"] + "\n\n=== CONSOLIDATED FINANCIAL STATEMENTS (PART IV) ===\n\n" + financials_part4
                logger.success(f"Đã gộp thành công Part IV / Item 15 vào Item 8 cho {file_path.name}. Độ dài mới: {len(extracted_data['Item 8'])} ký tự.")
        # -----------------------------------------
            
        # Fallback: Nếu không tìm thấy các mốc section do định dạng HTML đặc biệt,
        for name in sec_names:
            if name not in extracted_data:
                logger.warning(f"Không tìm thấy {name} bằng regex. File: {file_path.name}")
                
        return extracted_data
