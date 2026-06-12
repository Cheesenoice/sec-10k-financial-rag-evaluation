"""
config.py - Central configuration loader
Đọc từ .env file, cung cấp config cho toàn bộ project
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file từ root directory
load_dotenv()

# ─── Project Paths ────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent

DATA_RAW_DIR = ROOT_DIR / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR = ROOT_DIR / os.getenv("DATA_PROCESSED_DIR", "data/processed")
DATA_INDEXES_DIR = ROOT_DIR / os.getenv("DATA_INDEXES_DIR", "data/indexes")
DATA_EVAL_DIR = ROOT_DIR / os.getenv("DATA_EVAL_DIR", "data/eval")
EVAL_RESULTS_DIR = ROOT_DIR / os.getenv("EVAL_RESULTS_DIR", "eval/results")
EVAL_FIGURES_DIR = ROOT_DIR / os.getenv("EVAL_FIGURES_DIR", "eval/figures")
MODELS_DIR = ROOT_DIR / "models"

# Thiết lập biến môi trường để ép HuggingFace lưu model tải về vào thư mục models/ của dự án
os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")

# ─── API Keys ─────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ─── Model Names ──────────────────────────────────────────────
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ─── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE_DEFAULT = int(os.getenv("CHUNK_SIZE_DEFAULT", 512))
CHUNK_OVERLAP_DEFAULT = int(os.getenv("CHUNK_OVERLAP_DEFAULT", 64))
CHUNK_SIZE_SMALL = int(os.getenv("CHUNK_SIZE_SMALL", 256))   # Risk Factors

# ─── Retrieval ────────────────────────────────────────────────
BM25_TOP_K = int(os.getenv("BM25_TOP_K", 20))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", 20))
RRF_K = int(os.getenv("RRF_K", 60))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 5))

# ─── SEC EDGAR ────────────────────────────────────────────────
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "NLP-Student nlp@student.edu.vn")

# ─── Companies & Years ────────────────────────────────────────
# 6 companies × 3 years = 18 filings
TARGET_TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "GOOGL"]
TARGET_YEARS = [2022, 2023, 2024]

# ─── SEC Sections để parse ────────────────────────────────────
# Item name → (chunk_size, regex patterns to detect)
SEC_SECTIONS = {
    "Item 1": {
        "name": "Business",
        "chunk_size": CHUNK_SIZE_DEFAULT,
        "patterns": [r"item\s*1[.\s]*business", r"item\s*1\b"]
    },
    "Item 1A": {
        "name": "Risk Factors",
        "chunk_size": CHUNK_SIZE_SMALL,   # Dense → chunk nhỏ
        "patterns": [r"item\s*1a[.\s]*risk", r"risk\s*factors"]
    },
    "Item 7": {
        "name": "MD&A",
        "chunk_size": CHUNK_SIZE_DEFAULT,
        "patterns": [r"item\s*7[.\s]*management", r"management.{0,30}discussion"]
    },
    "Item 8": {
        "name": "Financial Statements",
        "chunk_size": CHUNK_SIZE_DEFAULT,
        "patterns": [r"item\s*8[.\s]*financial", r"financial\s*statements"]
    },
}
