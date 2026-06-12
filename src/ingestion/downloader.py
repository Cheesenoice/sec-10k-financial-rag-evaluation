"""
src/ingestion/downloader.py
Tải báo cáo 10-K từ SEC EDGAR API
Tuân thủ SEC User-Agent policy
"""

import os
import time
import requests
from pathlib import Path
from loguru import logger
from src.config import SEC_USER_AGENT, DATA_RAW_DIR

# Map Ticker -> CIK (Central Index Key) của SEC (phải padded đủ 10 chữ số)
TICKER_TO_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
    "GOOGL": "0001652044"
}

class SECDownloader:
    def __init__(self, user_agent: str = SEC_USER_AGENT, output_dir: Path = DATA_RAW_DIR):
        self.headers = {"User-Agent": user_agent}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check User-Agent hợp lệ (SEC yêu cầu định dạng: Name email@domain.com)
        if "YourName" in user_agent or "nlp@student" in user_agent or "@" not in user_agent:
            logger.warning(
                f"User-Agent '{user_agent}' có thể không hợp lệ. "
                "SEC EDGAR có thể chặn request. Vui lòng cập nhật trong .env"
            )

    def _get_submissions(self, cik: str) -> dict:
        """Lấy danh sách các filing từ SEC submissions endpoint"""
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Không thể lấy thông tin submissions cho CIK {cik}. Status code: {response.status_code}")
        # SEC limit: 10 requests / sec
        time.sleep(0.15)
        return response.json()

    def download_10k(self, ticker: str, year: int) -> Path:
        """Tải file 10-K của ticker trong năm chỉ định"""
        ticker = ticker.upper()
        if ticker not in TICKER_TO_CIK:
            raise ValueError(f"Không tìm thấy CIK cho ticker: {ticker}")
            
        cik = TICKER_TO_CIK[ticker]
        logger.info(f"Bắt đầu tìm báo cáo 10-K cho {ticker} (Năm tài chính: {year})...")
        
        data = self._get_submissions(cik)
        filings = data.get("filings", {}).get("recent", {})
        
        # Duyệt qua danh sách filings để tìm form 10-K
        accession_number = None
        primary_document = None
        report_date = None
        
        # Helper function to find in a filings dictionary
        def find_in_filings(f_dict):
            nonlocal accession_number, primary_document, report_date
            for idx, form in enumerate(f_dict.get("form", [])):
                if form == "10-K":
                    date_str = f_dict.get("reportDate", [])[idx]
                    filing_year = int(date_str.split("-")[0])
                    if filing_year == year:
                        accession_number = f_dict.get("accessionNumber", [])[idx]
                        primary_document = f_dict.get("primaryDocument", [])[idx]
                        report_date = date_str
                        return True
            # Fallback
            for idx, form in enumerate(f_dict.get("form", [])):
                if form == "10-K":
                    date_str = f_dict.get("filingDate", [])[idx]
                    filing_year = int(date_str.split("-")[0])
                    if filing_year == year or filing_year == year + 1:
                        accession_number = f_dict.get("accessionNumber", [])[idx]
                        primary_document = f_dict.get("primaryDocument", [])[idx]
                        report_date = f_dict.get("reportDate", [])[idx]
                        logger.info(f"Dùng fallback matching: nộp ngày {date_str} cho năm tài chính {year}")
                        return True
            return False

        # 1. Tìm trong filings recent trước
        found = find_in_filings(filings)

        # 2. Nếu chưa tìm thấy, duyệt qua các file lưu trữ lịch sử của SEC
        if not found:
            historical_files = data.get("filings", {}).get("files", [])
            logger.info(f"Không tìm thấy trong recent filings. Đang quét {len(historical_files)} file lưu trữ lịch sử...")
            for file_info in historical_files:
                file_name = file_info.get("name")
                if not file_name:
                    continue
                hist_url = f"https://data.sec.gov/submissions/{file_name}"
                logger.debug(f"Đang quét file lịch sử: {file_name}")
                hist_response = requests.get(hist_url, headers=self.headers)
                time.sleep(0.15)
                if hist_response.status_code == 200:
                    hist_data = hist_response.json()
                    if find_in_filings(hist_data):
                        found = True
                        break
                if found:
                    break

        if not accession_number or not primary_document:
            raise FileNotFoundError(f"Không tìm thấy báo cáo 10-K cho {ticker} năm {year}")

        # Tạo URL tải file 10-K gốc
        # Định dạng URL: https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_no_hyphens}/{primary_document}
        acc_no_hyphens = accession_number.replace("-", "")
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_hyphens}/{primary_document}"
        
        logger.info(f"Đang tải {ticker} 10-K từ: {doc_url}")
        
        response = requests.get(doc_url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Tải thất bại. Status code: {response.status_code}")
            
        # Lưu file
        file_ext = Path(primary_document).suffix  # Thường là .htm hoặc .txt
        output_file = self.output_dir / f"{ticker}_{year}_10-K{file_ext}"
        
        with open(output_file, "wb") as f:
            f.write(response.content)
            
        logger.success(f"Đã lưu: {output_file.name} ({len(response.content) / 1024 / 1024:.2f} MB)")
        time.sleep(0.15)
        return output_file
