# Kịch Bản Thuyết Trình Nâng Cao & Bộ 5 Câu Hỏi Demo RAG Chuyên Sâu

Tài liệu này cung cấp kịch bản thuyết trình học thuật nâng cao cùng **5 câu hỏi kiểm thử chuyên sâu** để chứng minh toàn diện năng lực của hệ thống RAG chatbot trước hội đồng phản biện.

---

## 🎭 PHẦN 1: BỘ 5 CÂU HỎI DEMO NÂNG CAO (ADVANCED SCENARIOS)

Hãy dùng các câu hỏi sau để chạy live và giải thích thuật toán tương ứng:

### 📅 Câu hỏi 1: Lỗi Nhiễu Thời Gian & Định Tuyến (Temporal Routing)
*   **Query:** `In fiscal year 2023, what were the constant currency revenues of GOOGL?`
*   **Mục tiêu kiểm thử:** Định tuyến chính xác năm tài chính 2023 và Ticker GOOGL trong cơ sở dữ liệu có 18 báo cáo tài chính giống hệt nhau về mặt cấu trúc.
*   **BM25/Dense Baseline (Lỗi):** Dễ bị nhiễu do trả về bảng doanh thu năm 2022 hoặc 2024 của GOOGL hoặc của hãng khác vì cụm từ `"constant currency revenues"` xuất hiện ở nhiều năm.
*   **Enhanced RAG (Đúng):** Thu hồi chính xác doanh thu ngoại tệ không đổi của GOOGL năm 2023.
*   **Giải thích thuật toán:** *NLP Year Routing* quét từ khóa `2023` và `GOOGL`, áp bộ lọc cứng (`metadata.year == 2023` và `metadata.ticker == 'GOOGL'`), cô lập không gian tìm kiếm từ 1931 chunks xuống còn ~50 chunks trước khi tính tương đồng.

---

### 💼 Câu hỏi 2: Khoảng Cách Từ Vựng Cực Hạn (Lexical Gap / Synonym)
*   **Query:** `What was the amount of capex for AMZN in 2023?`
*   **Mục tiêu kiểm thử:** Giải quyết viết tắt thuật ngữ tài chính (`capex` $\to$ Capital Expenditures) mà Amazon không ghi trực tiếp trong báo cáo.
*   **BM25 Baseline (Lỗi):** Báo lỗi không tìm thấy hoặc trả về kết quả không liên quan do từ khóa viết tắt `capex` không khớp với báo cáo 10-K của Amazon (hãng ghi là `"purchases of property and equipment"` trong Consolidated Statements of Cash Flows).
*   **Enhanced RAG (Đúng):** Trích xuất chính xác con số **$52,729 million**.
*   **Giải thích thuật toán:** *Query Expansion* tự động phát hiện từ `capex` và mở rộng câu hỏi thành: `"What was the amount of capex for AMZN in 2023? purchases of property and equipment acquisition of property plant and equipment capital spending"`.

---

### 📊 Câu hỏi 3: So Sánh Liên Công Ty (Multi-Company Comparison)
*   **Query:** `Compare the R&D spend of TSLA and MSFT in fiscal year 2023.`
*   **Mục tiêu kiểm thử:** Thu hồi đa tài liệu (Multi-Document Retrieval) song song cho 2 công ty khác nhau trong cùng một câu hỏi.
*   **Baselines (Lỗi):** Thường chỉ tập trung lấy thông tin của một trong hai công ty (hoặc chỉ TSLA hoặc chỉ MSFT) do giới hạn độ tương đồng của một vector truy vấn đơn lẻ.
*   **Enhanced RAG (Đúng):** Trích xuất đầy đủ và so sánh: R&D của MSFT (**$27,195 million**) và TSLA (**$3,969 million**).
*   **Giải thích thuật toán:** 
  1. *NLP Router* kích hoạt bộ lọc song song cho cả `TSLA` và `MSFT`.
  2. *Hybrid Search (RRF)* xếp hạng cao các tài liệu chứa số liệu R&D từ cả hai nguồn.
  3. LLM tổng hợp thông tin từ ngữ cảnh đa nguồn để đưa ra bảng so sánh trực quan.

---

### 🧩 Câu hỏi 4: Thuật Ngữ Kế Toán Phức Tạp (VIEs Accounting Test)
*   **Query:** `At the end of fiscal year 2023, what was the total carrying value of assets held by VIEs, as presented in the consolidated balance sheet of TSLA?`
*   **Mục tiêu kiểm thử:** Đọc hiểu các ghi chú kế toán phức tạp về VIEs (Variable Interest Entities - Thực thể có quyền lợi biến đổi).
*   **Baselines (Lỗi):** Trả về bảng cân đối kế toán chính (Consolidated Balance Sheet) vốn không liệt kê chi tiết tài sản của VIEs, hoặc trả về VIEs của năm khác.
*   **Enhanced RAG (Đúng):** Trả về chính xác giá trị ghi sổ là **$204 million** (nằm sâu trong Thuyết minh Note 16 - Variable Interest Entity Arrangements).
*   **Giải thích thuật toán:** *Cross-Encoder Reranker* nhận diện được sự tương quan ngữ nghĩa sâu sắc của Note 16 (dù từ khóa VIEs xuất hiện rất ít) và đẩy nó lên Top-1 vượt qua các bảng cân đối kế toán chung chung.

---

### 🎯 Câu hỏi 5: Số Liệu Chi Tiết Phân Khúc (Segment Analysis Analysis)
*   **Query:** `For GOOGL in 2023, what was the dollar value of the SBC expense that contributed to the increase in compensation expenses of Google Services?`
*   **Mục tiêu kiểm thử:** Trích xuất chi phí SBC (Share-Based Compensation - Bồi thường bằng cổ phiếu) đóng góp vào tăng chi phí nhân sự của phân khúc Google Services.
*   **Baselines (Lỗi):** Lấy nhầm tổng chi phí SBC toàn tập đoàn Google hoặc nhầm của phân khúc Google Cloud.
*   **Enhanced RAG (Đúng):** Trích xuất chính xác chi phí SBC đóng góp vào Google Services.
*   **Giải thích thuật toán:** Tìm kiếm lai (Hybrid Search) kết hợp thế mạnh khớp từ khóa chính xác phân khúc `"Google Services"` của BM25 và ngữ nghĩa ngữ cảnh `"SBC expense contribution"` của Dense HNSW.

---

## 🎤 PHẦN 2: KỊCH BẢN THUYẾT TRÌNH (PRESENTATION SCRIPT)

### ⏱️ 0:00 - 1:30 | Giới thiệu dự án & Cấu trúc chỉ mục
*   *"Kính thưa Hội đồng, hệ thống RAG của chúng em được thiết kế để giải quyết bài toán truy vấn tài chính trên **18 báo cáo SEC 10-K** với tổng quy mô **1931 chunks** văn bản phức tạp."*
*   *"Điểm đặc biệt của hệ thống là khả năng biểu diễn học thuật thông qua **3 chế độ Pipeline** độc lập trên giao diện Streamlit: **BM25 Lexical Baseline**, **Dense HNSW Baseline**, và **Enhanced RAG Pipeline**."*

### ⏱️ 1:30 - 3:30 | Trình diễn lỗi Baseline & Giải pháp Enhanced
*   **Biểu diễn Câu hỏi 1 (Temporal Routing):**
    *   *Hành động:* Chọn **Baseline 2 (Dense)**, chạy câu hỏi 1 $\to$ Nhận diện sai năm 2024. Chọn **Enhanced RAG**, chạy lại $\to$ Trả lời đúng.
    *   *Thuyết trình:* *"Dense vector bị pha loãng thông tin thời gian. Hệ thống cải tiến của chúng em sử dụng **NLP Year Routing** để trích xuất số năm và áp bộ lọc cứng siêu dữ liệu (Metadata Filter), loại bỏ 100% nhiễu thời gian."*
*   **Biểu diễn Câu hỏi 2 (Lexical Gap):**
    *   *Hành động:* Chọn **Baseline 1 (BM25)**, chạy câu hỏi 2 $\to$ Thất bại. Chọn **Enhanced RAG**, chạy lại $\to$ Thành công.
    *   *Thuyết trình:* *"BM25 không hiểu các từ viết tắt tài chính như 'capex'. Bằng cách áp dụng **Query Expansion**, hệ thống tự động bổ sung các cụm từ đồng nghĩa để vượt qua khoảng cách từ vựng."*

### ⏱️ 3:30 - 5:00 | Thuyết minh thuật toán qua Live Debug Panel
*   *Hành động:* Bấm mở rộng **Live Observability & Debug Panel** dưới câu trả lời của Enhanced RAG.
*   *Thuyết trình:* 
    *   *"Hệ thống của chúng em cung cấp tính năng **White-box Observability**. Thầy cô có thể thấy live ma trận gộp điểm **RRF (Reciprocal Rank Fusion)** gộp thứ hạng BM25 và Dense."*
    *   *"Tiếp theo, bảng **Cross-Encoder Reranker** hiển thị điểm tương quan chéo (Logits) chi tiết, chứng minh thuật toán xếp hạng chéo tối ưu hóa độ chính xác trước khi gửi ngữ cảnh tới LLM Llama 3.3."*
*   *Kết luận:* *"Nhờ sự kết hợp này, Recall@5 của hệ thống tăng từ **0.2875** (TF-IDF) và **0.4562** (BM25) lên mức **0.7719** đối với Enhanced RAG."*

---
