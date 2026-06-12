# SEC 10-K RAG QA Chatbot System

Hệ thống Hỏi đáp (Question-Answering Chatbot) ứng dụng mô hình RAG (Retrieval-Augmented Generation) tìm kiếm và phân tích báo cáo tài chính SEC 10-K của 6 tập đoàn công nghệ lớn (AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL) giai đoạn 2022 - 2024.

Dự án được xây dựng với mục tiêu kép:

1. **Production System:** Chạy ứng dụng thực tế với FastAPI Backend và Streamlit Frontend, hỗ trợ đối chiếu 3 pipeline tìm kiếm (Baseline 1, Baseline 2, Enhanced Pipeline).
2. **Academic Notebooks:** Bộ bài thực hành chi tiết giúp kiểm chứng công thức toán học và giải thích chi tiết các thuật toán phục vụ vấn đáp môn học NLP/IR.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
NLP-project/
├── data/                     # Thư mục chứa dữ liệu
│   ├── raw/                  # Báo cáo 10-K gốc dạng HTML từ SEC EDGAR
│   ├── processed/            # File documents.jsonl chứa các chunk đã parse
│   └── indexes/              # Chỉ mục nhị phân lưu trữ (FAISS .faiss + BM25 .pkl)
├── src/                      # Source code hệ thống RAG Production
│   ├── ingestion/            # Bộ nạp dữ liệu (downloader, parser, chunker)
│   ├── indexing/             # Bộ lập chỉ mục (BM25Index, VectorIndex HNSW)
│   ├── retrieval/            # Thuật toán tìm kiếm (Hybrid Search, CE Reranker)
│   ├── generation/           # Prompt Engineering & LLM clients (Groq/Ollama)
│   └── api/                  # FastAPI backend server
├── app/                      # Giao diện ứng dụng
│   └── streamlit_app.py      # Streamlit Frontend UI
├── notebooks/                # Bộ tài liệu Jupyer Notebook minh họa toán học
│   ├── preprocessing/        # Demo parsing và chunking thô
│   ├── baselines/            # Chạy thử nghiệm chi tiết các mức độ Baseline
│   │   ├── 1_lexical/        # Demo toán học TF-IDF và Okapi BM25
│   │   ├── 2_vector/         # Demo Dense Embeddings & trực quan đồ thị HNSW 2D
│   │   └── 3_enhanced/       # Mô phỏng luồng dữ liệu step-by-step hệ thống nâng cao
├── requirements.txt          # Các thư viện phụ thuộc
└── README.md                 # Hướng dẫn dự án này
```

---

## 🛠️ Chi Tiết Mã Nguồn Sản Phẩm (`src/`)

Hệ thống backend được chia thành các lớp chức năng độc lập theo mô hình RAG tiêu chuẩn:

| Tên File                            | Chức năng chính                                                                         | Vai trò trong hệ thống RAG         |
| :---------------------------------- | :-------------------------------------------------------------------------------------- | :--------------------------------- |
| `src/ingestion/downloader.py`       | Tải tự động các file 10-K từ cổng SEC EDGAR API.                                        | Thu thập dữ liệu thô (Raw Data).   |
| `src/ingestion/parser.py`           | Sử dụng BeautifulSoup để bóc tách mã HTML rác, trích xuất cấu trúc văn bản.             | Tiền xử lý dữ liệu (Parsing).      |
| `src/ingestion/chunker.py`          | Phân mảnh văn bản theo kích thước chỉ định (256/512 tokens) kèm sliding window overlap. | Phân mảnh ngữ cảnh (Chunking).     |
| `src/indexing/bm25_index.py`        | Lập chỉ mục từ khóa sử dụng thuật toán Okapi BM25.                                      | Lập chỉ mục Lexical Index.         |
| `src/indexing/vector_index.py`      | Sử dụng `bge-small-en-v1.5` sinh dense vector và dựng đồ thị FAISS HNSW.                | Lập chỉ mục Semantic Index.        |
| `src/retrieval/hybrid_retriever.py` | Chạy truy vấn song song BM25 + HNSW và gộp điểm bằng thuật toán RRF.                    | Tìm kiếm lai (Hybrid Retrieval).   |
| `src/retrieval/reranker.py`         | Sử dụng mô hình Cross-Encoder để tính điểm tương quan sâu giữa query và doc.            | Tái xếp hạng (Reranking).          |
| `src/generation/llm_client.py`      | Quản lý kết nối tới Groq Cloud API hoặc Ollama local chạy ngoại tuyến.                  | Sinh câu trả lời (LLM Generation). |
| `src/api/main.py`                   | Điểm kết nối FastAPI, xử lý NLP Router, Query Expansion và phân phối 3 pipeline.        | Cổng kết nối (API Gateway).        |

---

## 📓 Chi Tiết Bộ Notebooks Học Tập (`notebooks/`)

Mục đích chính của thư mục `notebooks/` là giúp bạn **học và vấn đáp trực quan**. Toàn bộ các notebook sử dụng một **Corpus Mẫu (5 câu tài chính)** để người học và giáo viên có thể tính toán thủ công từng bước:

### 1. `notebooks/preprocessing/` (Tiền xử lý)

- `1_parsing_demo.ipynb`: Hướng dẫn trích xuất các phân mục (Item 1A, Item 7, Item 8) bằng Regex trên tài liệu HTML.
- `2_chunking_demo.ipynb`: Trực quan hóa các chunk gối đầu nhau (Overlap) bằng kỹ thuật Sliding Window để không mất ngữ cảnh ở biên.

### 2. `notebooks/baselines/` (Mô hình tìm kiếm)

- `1_lexical/1a_demo_tfidf.ipynb`: Mô phỏng toán học Mô hình không gian Vector (VSM):
  - Tự xây dựng từ điển từ vựng (Vocabulary).
  - In ra **toàn bộ Ma trận TF**, **bảng IDF** và **Ma trận trọng số TF-IDF**.
  - Minh họa từng phép nhân của độ tương đồng **Cosine Similarity** và phân tích lỗi khoảng cách từ vựng (Lexical Gap).
- `1_lexical/1b_demo_bm25.ipynb`: Đi sâu vào thuật toán **Okapi BM25**:
  - Công thức bão hòa tần suất từ ($k_1$) và cơ chế phạt độ dài văn bản ($b$).
  - In **Ma trận trọng số BM25** đầy đủ giữa mọi tài liệu và từ vựng.
- `2_vector/2a_demo_vector.ipynb`: Chuyển sang tìm kiếm ngữ nghĩa Dense Vector:
  - Sinh vector biểu diễn 384 chiều bằng mô hình `bge-small-en-v1.5`.
  - In ma trận tương quan ngữ nghĩa $5 \times 5$ giữa các tài liệu.
  - **Trực quan hóa đồ thị HNSW 2D:** Sử dụng PCA giảm chiều xuống 2D, dùng NetworkX và Matplotlib vẽ cấu trúc liên kết đồ thị của FAISS.
  - Chỉ ra lỗi **Nhiễu Thời gian (Temporal Mismatch)** khi số liệu năm 2024 đè lên năm 2023.
- `3_enhanced/3a_demo_enhanced.ipynb`: Trình diễn luồng dữ liệu step-by-step cải tiến:
  - **NLP Year Routing:** Regex trích xuất năm và áp dụng bộ lọc cứng loại bỏ nhiễu thời gian.
  - **Query Expansion:** Mở rộng từ đồng nghĩa tự động.
  - **RRF Fusion:** Gộp thứ hạng BM25 & HNSW (in bảng phân rã phân số RRF).
  - **Cross-Encoder Reranking:** Đo độ tương quan sâu qua Attention chéo.

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

> [!NOTE]
> **Dữ liệu chỉ mục đã được dựng sẵn:** Các tệp cơ sở dữ liệu và chỉ mục tìm kiếm (BM25 `.pkl` và FAISS `.faiss` trong thư mục `data/indexes/`) đã được tính toán và đẩy kèm trong kho mã nguồn này. Bạn **không cần chạy lại bước xây dựng chỉ mục**, chỉ cần cấu hình khóa API Groq là có thể chạy thử nghiệm chatbot ngay lập tức.

### Yêu cầu hệ thống

- Python 3.10 trở lên.
- Groq API Key (Đăng ký miễn phí tại [Groq Console](https://console.groq.com)).

### 1. Khởi tạo môi trường ảo & Cài đặt thư viện

```powershell
# Tạo venv
python -m venv venv
# Kích hoạt venv trên Windows
venv\Scripts\activate
# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Cấu hình biến môi trường

Tạo file `.env` tại thư mục gốc của dự án với nội dung:

```env
GROQ_API_KEY=your_api_key_here
```

### 3. Chạy hệ thống (Chạy song song 2 Terminal)

**Terminal 1: Khởi chạy FastAPI Backend Server**

```powershell
$env:PYTHONUTF8=1
venv\Scripts\python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

_(Đợi log xuất hiện chữ: `API khởi động và nạp chỉ mục thành công!`)_

**Terminal 2: Khởi chạy Streamlit Frontend UI**

```powershell
venv\Scripts\streamlit run app/streamlit_app.py
```

_(Trình duyệt sẽ tự động mở trang giao diện tại địa chỉ `http://localhost:8501`)_

---

## 🧪 Kịch Bản Demo Vấn Đáp (Đối Chiếu Lỗi & Sửa Lỗi)

Khi thuyết trình trực tiếp cho giáo viên, hãy nhập 2 câu hỏi sau và đổi chế độ Pipeline trên Sidebar để đối chiếu:

### Câu 1: Lỗi khoảng cách từ vựng (Lexical Gap)

- **Query:** `What are Amazon's capital expenditures in 2023?`
  - _Sự thật:_ Báo cáo Amazon không dùng từ `"capital expenditures"` mà dùng `"purchases of property and equipment"`.
  - **Chạy Baseline 1 (BM25 Lexical):** Trả về điểm số thấp hoặc báo không tìm thấy thông tin vì lệch từ khóa thô.
  - **Chạy Enhanced RAG:** Tìm ra chính xác số liệu **$52,729 million** nhờ cơ chế **Query Expansion** tự động dịch cụm từ đồng nghĩa.

### Câu 2: Lỗi nhiễu thời gian (Temporal Mismatch)

- **Query:** `What was NVIDIA's net income in 2023?`
  - _Sự thật:_ Dữ liệu mẫu chỉ nạp doanh thu NVIDIA năm **2024** ($29,760 million). Không có năm **2023**.
  - **Chạy Baseline 2 (Dense Vector HNSW):** Trả về số liệu của năm **2024** vì vector ngữ nghĩa `"NVIDIA net income"` tương đồng cao, bỏ qua chữ số năm.
  - **Chạy Enhanced RAG:** Trả lời chính xác **Không tìm thấy thông tin** (hoặc trích dẫn đúng năm 2023 từ bảng đối chiếu nếu có) vì **NLP Year Routing** đã trích xuất số `2023` và lọc cứng loại bỏ tài liệu năm 2024 ngay từ bước đầu.
