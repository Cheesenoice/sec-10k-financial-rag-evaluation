"""
app/streamlit_app.py
Streamlit Frontend Chat UI for SEC 10-K RAG QA System
Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import requests
import json
import pandas as pd

# Configure Streamlit Page
st.set_page_config(
    page_title="SEC RAG Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint of FastAPI Backend
API_URL = "http://localhost:8000/query"

# Application Title
st.title("💼 SEC 10-K RAG QA System")
st.markdown("___")

# ─── SIDEBAR: Filters & Configurations ─────────────────────────────
st.sidebar.header("⚙️ RAG Configuration")

# Pipeline Mode Configuration
pipeline_mode = st.sidebar.selectbox(
    "🤖 Pipeline Mode",
    options=["enhanced_pipeline", "baseline_1_lexical", "baseline_2_semantic"],
    format_func=lambda x: {
        "enhanced_pipeline": "Enhanced RAG Pipeline",
        "baseline_1_lexical": "Baseline 1 (BM25 Lexical)",
        "baseline_2_semantic": "Baseline 2 (Dense Vector HNSW)"
    }[x],
    help="Select the RAG pipeline mode to test, compare, and observe performance."
)

# LLM Selection Toggle
use_local = st.sidebar.toggle("🖥️ Use Local LLM (Ollama)", value=False)
model_source = "Ollama Local (Llama 3.2)" if use_local else "Groq Cloud (Llama 3.3)"
st.sidebar.caption(f"Active LLM Engine: **{model_source}**")

# Retrieval Filters (Sidebar Filter)
st.sidebar.subheader("🔍 Hard Filters")
st.sidebar.caption("Note: NLP Router automatically extracts entities from your query, taking precedence over sidebar filters.")

tickers_list = ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "GOOGL"]
years_list = [2022, 2023, 2024]

selected_tickers = st.sidebar.multiselect(
    "Companies (Tickers)",
    options=tickers_list,
    default=tickers_list,
    help="Restrict the search to the selected stock tickers."
)

selected_years = st.sidebar.multiselect(
    "Fiscal Years",
    options=years_list,
    default=years_list,
    help="Restrict the search to the selected filing years."
)

top_k_chunks = st.sidebar.slider(
    "Top-K Chunks to Retrieve",
    min_value=1,
    max_value=10,
    value=5,
    help="Number of most relevant chunks passed into the LLM context."
)

st.sidebar.markdown("---")
st.sidebar.caption("NLP Course Project - 2026")

# ─── Chat Session State Initializer ──────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Default Welcome Message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I am your SEC 10-K Financial Analyst Assistant. Ask me factual or comparative questions about the financial data of AAPL, MSFT, AMZN, NVDA, TSLA, or GOOGL.",
        "citations": [],
        "debug_info": None
    })

# Helper function to render the live observability panel
def render_debug_panel(db):
    if not db:
        return
    
    st.markdown("### 🛠️ Stage 1: Query Processing & Routing")
    col1, col2, col3 = st.columns(3)
    col1.metric("Pipeline Mode", db["pipeline_mode"])
    
    # Display Ticker / Year routing
    tickers_detected = db.get("detected_tickers", [])
    years_detected = db.get("detected_years", [])
    
    col2.markdown(f"**🏷️ NLP Tickers Router:** {', '.join(tickers_detected) if tickers_detected else 'None detected (Using default filters)'}")
    col3.markdown(f"**📅 NLP Years Router:** {', '.join(map(str, years_detected)) if years_detected else 'None detected (Using default filters)'}")
    
    # Display Query Expansion
    if db.get("expanded_query"):
        st.info(f"**📝 Query Expansion (Synonyms):** '{db['expanded_query']}'")
        
    st.markdown("___")
    st.markdown("### 🔄 Stage 2: Parallel Candidate Retrieval & Fusion")
    
    tab1, tab2, tab3 = st.tabs(["BM25 Candidates", "Vector Candidates", "RRF Fusion Matrix"])
    
    with tab1:
        if db.get("bm25_raw_results"):
            bm25_df = pd.DataFrame(db["bm25_raw_results"])
            st.dataframe(bm25_df, use_container_width=True)
        else:
            st.write("BM25 was not executed or returned empty results.")
            
    with tab2:
        if db.get("vector_raw_results"):
            vec_df = pd.DataFrame(db["vector_raw_results"])
            st.dataframe(vec_df, use_container_width=True)
        else:
            st.write("Vector HNSW search was not executed or returned empty results.")
            
    with tab3:
        if db.get("rrf_details"):
            rrf_df = pd.DataFrame(db["rrf_details"])
            # Format DataFrame columns
            rrf_df.columns = ["Chunk ID", "Ticker", "Year", "Section", "BM25 Rank", "BM25 RRF Score", "Vector Rank", "Vector RRF Score", "Total RRF Score"]
            st.dataframe(rrf_df, use_container_width=True)
        else:
            st.write("RRF Fusion ranking was not applied in this pipeline mode.")
            
    # Display Cross-Encoder Reranker
    if db.get("reranked_results"):
        st.markdown("___")
        st.markdown("### 🎯 Stage 3: Deep Relevance Reranking (Cross-Encoder)")
        st.caption("Candidates cross-evaluated using Self-Attention to re-score precise semantic relevance logits:")
        ce_df = pd.DataFrame(db["reranked_results"])
        ce_df.columns = ["Chunk ID", "Ticker", "Year", "Reranker Score (Logits)"]
        st.dataframe(ce_df, use_container_width=True)

# ─── Render Chat History ──────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Display Citations Dropdown
        if msg.get("citations"):
            with st.expander("📚 Citations (Source Documents)"):
                for idx, cite in enumerate(msg["citations"]):
                    st.markdown(
                        f"**[{idx+1}] {cite['ticker']} ({cite['year']}) - {cite['section']}** "
                        f"*(Score: {cite['score']:.4f})* | `Source: {cite.get('source', 'unknown')}` | `ID: {cite['chunk_id']}`"
                    )
                    st.caption(f"\"{cite['text']}\"")
                    st.markdown("---")
                    
        # Display Observability panel for the message
        if msg.get("debug_info"):
            with st.expander("🛠️ Live Observability & Debug Panel"):
                render_debug_panel(msg["debug_info"])

# ─── Read User Question from Input ───────────────────────
if prompt := st.chat_input("Enter your question (e.g., What was Amazon's capital expenditures in 2023?)..."):
    
    # 1. Display User Message
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Call FastAPI backend API
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("*Processing (Retrieval + Rerank + LLM)...*")
        
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
                
                # Update generated answer
                response_placeholder.write(answer)
                
                # Display Citations dropdown
                if citations:
                    with st.expander("📚 Citations (Source Documents)"):
                        for idx, cite in enumerate(citations):
                            st.markdown(
                                f"**[{idx+1}] {cite['ticker']} ({cite['year']}) - {cite['section']}** "
                                f"*(Score: {cite['score']:.4f})* | `Source: {cite.get('source', 'unknown')}` | `ID: {cite['chunk_id']}`"
                            )
                            st.caption(f"\"{cite['text']}\"")
                            st.markdown("---")
                            
                # Display Debug info dropdown
                with st.expander("🛠️ Live Observability & Debug Panel"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Retrieval Latency", f"{data['retrieval_latency_ms']} ms")
                    col2.metric("LLM Latency", f"{data['generation_latency_ms']} ms")
                    col3.metric("LLM Provider", data["llm_source"])
                    render_debug_panel(debug_info)
                    
                # Save to session history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "debug_info": debug_info
                })
                
            else:
                err_detail = res.json().get("detail", "Unknown error")
                response_placeholder.error(f"System Error: {err_detail} (Status code: {res.status_code})")
                
        except requests.exceptions.ConnectionError:
            response_placeholder.error(
                "Could not connect to FastAPI Backend API (port 8000). "
                "Please make sure the backend server is running: `uvicorn src.api.main:app --reload`"
                "\nIf uvicorn crashed due to Groq API Rate Limits, please switch on Ollama local toggle."
            )
        except Exception as e:
            response_placeholder.error(f"An error occurred: {e}")
