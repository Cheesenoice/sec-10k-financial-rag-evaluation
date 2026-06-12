"""
app/streamlit_app.py
Streamlit Frontend Chat UI cho SEC 10-K RAG QA System
Chạy: streamlit run app/streamlit_app.py
"""

import streamlit as st
import requests
import json
import pandas as pd

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="SEC RAG Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint của FastAPI
API_URL = "http://localhost:8000/query"

# Tiêu đề ứng dụng
st.title("💼 SEC 10-K RAG QA System")
st.markdown("___")

# ─── SIDEBAR: Bộ lọc và cấu hình ─────────────────────────────
st.sidebar.header("⚙️ Cấu hình RAG")

# Cấu hình Pipeline Mode
pipeline_mode = st.sidebar.selectbox(
    "🤖 Chế độ Pipeline",
    options=["enhanced_pipeline", "baseline_1_lexical", "baseline_2_semantic"],
    format_func=lambda x: {
        "enhanced_pipeline": "Hệ thống cải tiến (Enhanced RAG)",
        "baseline_1_lexical": "Baseline 1 (BM25 Lexical)",
        "baseline_2_semantic": "Baseline 2 (Dense Vector HNSW)"
    }[x],
    help="Chọn phiên bản RAG để chạy đối chiếu và làm rõ điểm yếu/mạnh"
)

# Cấu hình LLM
use_local = st.sidebar.toggle("🖥️ Sử dụng LLM Local (Ollama)", value=False)
model_source = "Ollama Local (Llama 3.2)" if use_local else "Groq Cloud (Llama 3.3)"
st.sidebar.caption(f"LLM hiện tại: **{model_source}**")

# Cấu hình Retrieval
st.sidebar.subheader("🔍 Bộ lọc thủ công (Sidebar Filter)")
st.sidebar.caption("Lưu ý: Bộ lọc tự động của Router sẽ ưu tiên hơn nếu phát hiện thực thể trong câu hỏi.")

tickers_list = ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "GOOGL"]
years_list = [2022, 2023, 2024]

selected_tickers = st.sidebar.multiselect(
    "Mã cổ phiếu (Tickers)",
    options=tickers_list,
    default=tickers_list,
    help="Chỉ tìm kiếm trong các báo cáo của mã cổ phiếu đã chọn"
)

selected_years = st.sidebar.multiselect(
    "Năm tài chính",
    options=years_list,
    default=years_list,
    help="Chỉ tìm kiếm trong các báo cáo nộp của năm đã chọn"
)

top_k_chunks = st.sidebar.slider(
    "Số lượng chunk trích xuất (Top-K Chunks)",
    min_value=1,
    max_value=10,
    value=5,
    help="Số lượng văn bản liên quan nhất được chuyển vào ngữ cảnh của LLM"
)

st.sidebar.markdown("---")
st.sidebar.caption("NLP Course Project - 2026")

# ─── Khởi tạo Session State cho Lịch sử Chat ────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Thêm câu chào mặc định
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Xin chào! Tôi là Trợ lý Phân tích Tài chính SEC 10-K. Hãy đặt các câu hỏi so sánh hoặc tìm kiếm số liệu tài chính của AAPL, MSFT, AMZN, NVDA, TSLA hoặc GOOGL.",
        "citations": [],
        "debug_info": None
    })

# Helper function để hiển thị debug panel
def render_debug_panel(db):
    if not db:
        return
    
    st.markdown("### 🛠️ Giai Đoạn 1: Xử Lý Truy Vấn & Định Tuyến (Query Routing)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Pipeline Mode", db["pipeline_mode"])
    
    # Hiển thị Ticker / Year routing
    tickers_detected = db.get("detected_tickers", [])
    years_detected = db.get("detected_years", [])
    
    col2.markdown(f"**🏷️ NLP Tickers Router:** {', '.join(tickers_detected) if tickers_detected else 'Không phát hiện (Dùng bộ lọc mặc định)'}")
    col3.markdown(f"**📅 NLP Years Router:** {', '.join(map(str, years_detected)) if years_detected else 'Không phát hiện (Dùng bộ lọc mặc định)'}")
    
    # Hiển thị Query Expansion
    if db.get("expanded_query"):
        st.info(f"**📝 Query Expansion (Mở rộng từ khóa):** '{db['expanded_query']}'")
        
    st.markdown("___")
    st.markdown("### 🔄 Giai Đoạn 2: Kết Quả Thu Hồi Từ Các Chỉ Mục (Parallel Retrieval)")
    
    tab1, tab2, tab3 = st.tabs(["BM25 Candidates", "Vector Candidates", "RRF Fusion Matrix"])
    
    with tab1:
        if db.get("bm25_raw_results"):
            bm25_df = pd.DataFrame(db["bm25_raw_results"])
            st.dataframe(bm25_df, use_container_width=True)
        else:
            st.write("Không chạy hoặc không có kết quả từ BM25.")
            
    with tab2:
        if db.get("vector_raw_results"):
            vec_df = pd.DataFrame(db["vector_raw_results"])
            st.dataframe(vec_df, use_container_width=True)
        else:
            st.write("Không chạy hoặc không có kết quả từ Vector HNSW.")
            
    with tab3:
        if db.get("rrf_details"):
            # Chuyển đổi list of dicts thành DataFrame
            rrf_df = pd.DataFrame(db["rrf_details"])
            # Format hiển thị
            rrf_df.columns = ["Chunk ID", "Ticker", "Năm", "Phần", "Hạng BM25", "Điểm RRF BM25", "Hạng Vector", "Điểm RRF Vector", "Tổng Điểm RRF"]
            st.dataframe(rrf_df, use_container_width=True)
        else:
            st.write("Không áp dụng gộp thứ hạng RRF ở chế độ này.")
            
    # Hiển thị Cross-Encoder Reranker
    if db.get("reranked_results"):
        st.markdown("___")
        st.markdown("### 🎯 Giai Đoạn 3: Tái Xếp Hạng Đọc Hiểu Sâu (Cross-Encoder Reranker)")
        st.caption("Các ứng viên được so khớp chéo Self-Attention để chấm điểm lại mức độ liên quan ngữ nghĩa chi tiết:")
        ce_df = pd.DataFrame(db["reranked_results"])
        ce_df.columns = ["Chunk ID", "Ticker", "Năm", "Điểm Reranker (Logits)"]
        st.dataframe(ce_df, use_container_width=True)

# ─── Hiển thị Lịch sử Chat ──────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Hiển thị Citations nếu có
        if msg.get("citations"):
            with st.expander("📚 Nguồn trích dẫn gốc (Citations)"):
                for idx, cite in enumerate(msg["citations"]):
                    st.markdown(
                        f"**[{idx+1}] {cite['ticker']} ({cite['year']}) - {cite['section']}** "
                        f"*(Score: {cite['score']:.4f})* | `Nguồn: {cite.get('source', 'unknown')}` | `ID: {cite['chunk_id']}`"
                    )
                    st.caption(f"\"{cite['text']}\"")
                    st.markdown("---")
                    
        # Hiển thị Debug Panel của từng tin nhắn nếu có
        if msg.get("debug_info"):
            with st.expander("🛠️ Live Observability & Debug Panel"):
                render_debug_panel(msg["debug_info"])

# ─── Tiếp nhận Câu hỏi từ Chat Input ───────────────────────
if prompt := st.chat_input("Nhập câu hỏi của bạn (ví dụ: What is Amazon's capital expenditures in 2023?)..."):
    
    # 1. Hiển thị câu hỏi của user
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Gọi API FastAPI
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("*Đang xử lý (Retrieval + Rerank + LLM)...*")
        
        payload = {
            "query": prompt,
            "top_k": top_k_chunks,
            "use_local_llm": use_local,
            "pipeline_mode": pipeline_mode,
            "filter_tickers": selected_tickers,
            "filter_years": selected_years
        }
        
        try:
            res = requests.post(API_URL, json=payload, timeout=60)
            
            if res.status_code == 200:
                data = res.json()
                answer = data["answer"]
                citations = data["citations"]
                debug_info = data["debug_info"]
                
                # Update câu trả lời
                response_placeholder.write(answer)
                
                # Hiển thị Citations dropdown
                if citations:
                    with st.expander("📚 Nguồn trích dẫn gốc (Citations)"):
                        for idx, cite in enumerate(citations):
                            st.markdown(
                                f"**[{idx+1}] {cite['ticker']} ({cite['year']}) - {cite['section']}** "
                                f"*(Score: {cite['score']:.4f})* | `Nguồn: {cite.get('source', 'unknown')}` | `ID: {cite['chunk_id']}`"
                            )
                            st.caption(f"\"{cite['text']}\"")
                            st.markdown("---")
                            
                # Hiển thị Debug info dropdown
                with st.expander("🛠️ Live Observability & Debug Panel"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Retrieval Latency", f"{data['retrieval_latency_ms']} ms")
                    col2.metric("LLM Latency", f"{data['generation_latency_ms']} ms")
                    col3.metric("LLM Provider", data["llm_source"])
                    render_debug_panel(debug_info)
                    
                # Lưu vào lịch sử chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "debug_info": debug_info
                })
                
            else:
                err_detail = res.json().get("detail", "Không rõ lỗi")
                response_placeholder.error(f"Lỗi hệ thống: {err_detail} (Status code: {res.status_code})")
                
        except requests.exceptions.ConnectionError:
            response_placeholder.error(
                "Không thể kết nối tới API Backend (cổng 8000). "
                "Vui lòng chạy lệnh khởi động API trước: `uvicorn src.api.main:app --reload`"
                "\nNếu uvicorn bị crash vì Rate Limit của Groq API, vui lòng bật cổng Ollama local."
            )
        except Exception as e:
            response_placeholder.error(f"Đã xảy ra lỗi: {e}")
