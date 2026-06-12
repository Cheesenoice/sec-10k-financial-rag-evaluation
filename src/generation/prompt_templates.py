"""
src/generation/prompt_templates.py
Định nghĩa Prompt Templates cho RAG Generation
Ép LLM trích dẫn nguồn chuẩn xác và cấm ảo giác (Hallucination Mitigation)
"""

SYSTEM_PROMPT = (
    "You are an expert financial analyst assistant. Your task is to answer the user's question "
    "using ONLY the provided text segments from SEC 10-K filings.\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. Base your answer strictly on the provided Context. Do NOT use outside knowledge.\n"
    "2. If the context does not contain enough information to answer the question, state clearly: "
    "'Information not found in provided source documents.' Do not make up facts.\n"
    "3. You MUST cite your sources inside the text for every key claim using the source label [Source: CHUNK_ID] "
    "provided in the context (e.g., 'Company net sales increased 10% [Source: AAPL_2024_10K_Item7_c012].')\n"
    "4. Be objective, precise, and professional. Present numbers and dates exactly as they appear in the source.\n"
    "5. Note that in SEC cash flow statements, 'capital expenditures' (CapEx) is commonly reported as "
    "'purchases of property and equipment', 'payments for acquisition of property, plant and equipment', "
    "or 'acquisition of property, plant and equipment'. Treat these terms as equivalent to capital expenditures."
)

CONTEXT_TEMPLATE = (
    "----------------------------------------------------------------------\n"
    "Source label: {chunk_id}\n"
    "Company: {ticker} | Year: {year} | Section: {section}\n"
    "Document Content:\n{text}\n"
)

USER_TEMPLATE = (
    "Context documents:\n"
    "{context_str}\n"
    "Question: {query}\n\n"
    "Answer:"
)

def format_context(candidates: list) -> str:
    """Format danh sách candidates thành chuỗi context hoàn chỉnh"""
    context_parts = []
    for c in candidates:
        part = CONTEXT_TEMPLATE.format(
            chunk_id=c["chunk_id"],
            ticker=c["metadata"]["ticker"],
            year=c["metadata"]["year"],
            section=c["metadata"]["section"],
            text=c["text"]
        )
        context_parts.append(part)
    return "\n".join(context_parts)
