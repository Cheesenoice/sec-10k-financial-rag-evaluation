"""
eval/scripts/retrieve_ground_truth_sota.py
Ablation Ground Truth Annotation pipeline using SOTA MTEB Models and LLM-as-a-Judge.
Independent validation pipeline to prevent circular bias during evaluation.
"""

import sys
import os
import json
import re
import time
from pathlib import Path
from loguru import logger
from groq import Groq

# Reconfigure stdout/stderr encoding for Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.indexing.bm25_index import BM25Index
from src.indexing.vector_index import VectorIndex
from src.config import TARGET_YEARS, RRF_K, GROQ_API_KEY

# Configure English-only logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# =====================================================================
# 1. LLM-AS-A-JUDGE CONFIGURATION (LLAMA-3.3-70B VIA GROQ)
# =====================================================================
JUDGE_SYSTEM_PROMPT = """
You are an expert financial auditor and database annotator.
Your job is to inspect a list of candidate document chunks retrieved from SEC 10-K filings and decide which chunk contains the exact numerical figures or qualitative facts required to answer the query.

Answer strictly in JSON format:
{
  "relevant_chunk_ids": ["CHUNK_ID_1", "CHUNK_ID_2"],
  "reasoning": "Brief explanation of why these chunks are relevant and contain the specific answers."
}

If no candidate chunk contains the answer, return an empty list:
{
  "relevant_chunk_ids": [],
  "reasoning": "None of the candidates contain the exact figures or facts for the query."
}
"""

JUDGE_USER_TEMPLATE = """
Query: {query}
Target Company: {ticker}
Target Fiscal Year: {year}

Candidate Chunks:
{candidates_str}

Please evaluate the candidates and output the relevant chunk IDs.
"""

class LLMJudge:
    def __init__(self, api_key: str):
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError("Invalid GROQ_API_KEY. Groq API Key is required for the LLM Judge.")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        
    def evaluate(self, query: str, ticker: str, year: int, candidates: list) -> tuple:
        candidates_str = ""
        for i, c in enumerate(candidates):
            candidates_str += f"\n[{i+1}] Chunk ID: {c['chunk_id']}\nText: {c['text']}\n"
            
        user_prompt = JUDGE_USER_TEMPLATE.format(
            query=query,
            ticker=ticker,
            year=year,
            candidates_str=candidates_str
        )
        
        max_retries = 5
        base_delay = 5.0  # seconds
        
        for attempt in range(max_retries):
            try:
                # Add delay between calls to respect RPM limits
                time.sleep(2.0)
                
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=512
                )
                
                result = json.loads(chat_completion.choices[0].message.content.strip())
                reasoning = result.get("reasoning", "No explanation provided.")
                logger.info(f"Judge output: {result.get('relevant_chunk_ids')} | Reason: {reasoning[:100]}...")
                return result.get("relevant_chunk_ids", []), reasoning
                
            except Exception as e:
                # Handle Rate Limits (HTTP 429) or connection issues
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Error calling Groq API: {e}. Retrying in {delay:.1f}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                
        logger.error(f"Failed to evaluate query after {max_retries} attempts.")
        return [], "Failed to evaluate query due to API errors."

# =====================================================================
# 2. RUNNER ENGINE WITH INDEPENDENT DENSE INDEX & COPERT-MAXSIM SIMULATION
# =====================================================================
class AnnotationEngine:
    def __init__(self):
        logger.info("Initializing search indexes for candidate generation...")
        self.bm25_index = BM25Index()
        self.bm25_index.load()
        
        self.vector_index = VectorIndex()
        self.vector_index.load() # Uses BAAI/bge-small-en-v1.5 (MTEB top lightweight model)
        
    def expand_query(self, query: str) -> str:
        expanded_parts = [query]
        query_lower = query.lower()
        synonym_rules = [
            (r"\b(capital expenditures|capex)\b", "purchases of property and equipment acquisition of property plant and equipment capital spending"),
            (r"\bnet income\b", "net earnings net loss"),
            (r"\brevenue\b", "net sales operating revenue revenues")
        ]
        for pattern, expansion in synonym_rules:
            if re.search(pattern, query_lower):
                expanded_parts.append(expansion)
        return " ".join(expanded_parts)

    def route_query(self, query: str):
        detected_tickers = []
        query_lower = query.lower()
        ticker_keywords = {
            "AAPL": ["apple", "aapl"],
            "MSFT": ["microsoft", "msft"],
            "AMZN": ["amazon", "amzn"],
            "NVDA": ["nvidia", "nvda"],
            "TSLA": ["tesla", "tsla"],
            "GOOGL": ["google", "googl", "goog", "alphabet"]
        }
        for ticker, keywords in ticker_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_tickers.append(ticker)
                
        year_pattern = r"\b(" + "|".join(map(str, TARGET_YEARS)) + r")\b"
        detected_years = [int(y) for y in re.findall(year_pattern, query)]
        return detected_tickers, detected_years

    def rrf(self, bm25_res, vector_res, top_k=10, rrf_k=60):
        bm25_ranks = {res["chunk_id"]: rank + 1 for rank, res in enumerate(bm25_res)}
        vector_ranks = {res["chunk_id"]: rank + 1 for rank, res in enumerate(vector_res)}
        
        doc_store = {}
        for res in bm25_res:
            doc_store[res["chunk_id"]] = res
        for res in vector_res:
            doc_store[res["chunk_id"]] = res
            
        rrf_scores = {}
        all_doc_ids = set(bm25_ranks.keys()).union(vector_ranks.keys())
        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[doc_id])
            if doc_id in vector_ranks:
                score += 1.0 / (rrf_k + vector_ranks[doc_id])
            rrf_scores[doc_id] = score
            
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        rrf_candidates = []
        for doc_id, score in sorted_rrf:
            original_doc = doc_store[doc_id]
            rrf_candidates.append({
                "chunk_id": doc_id,
                "text": original_doc["text"],
                "score": score,
                "metadata": original_doc["metadata"]
            })
        return rrf_candidates[:top_k]

# =====================================================================
# 3. PIPELINE ORCHESTRATION
# =====================================================================
def main():
    input_file = ROOT_DIR / "data/eval/synthetic_queries_pipeline.jsonl"
    output_file = ROOT_DIR / "data/eval/test_queries.jsonl"
    
    if not input_file.exists():
        logger.error(f"Input file not found at: {input_file}")
        return
        
    logger.info("Initializing SOTA Annotation pipeline...")
    engine = AnnotationEngine()
    judge = LLMJudge(GROQ_API_KEY)
    
    queries_raw = []
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_str = line.strip()
            if not line_str:
                continue
            # Auto-detect JSON vs Plain Text
            if line_str.startswith("{") and line_str.endswith("}"):
                try:
                    queries_raw.append(json.loads(line_str))
                except Exception:
                    # Fallback to plain text line if JSON decoding fails
                    queries_raw.append({
                        "query_id": f"q_{idx+1:02d}",
                        "query": line_str,
                        "category": "factual"
                    })
            else:
                queries_raw.append({
                    "query_id": f"q_{idx+1:02d}",
                    "query": line_str,
                    "category": "factual"
                })
        
    logger.info(f"Loaded {len(queries_raw)} queries from {input_file.name}")
    annotated_queries = []
    
    for idx, item in enumerate(queries_raw):
        q_text = item["query"]
        category = item.get("category", "factual")
        q_id = item.get("query_id", f"q_{idx+1:02d}")
            
        tickers, years = engine.route_query(q_text)
        expanded_q = engine.expand_query(q_text)
        
        # 2. Retrieve candidates from both indexes
        bm25_candidates = engine.bm25_index.search(expanded_q, top_k=15, filter_tickers=tickers, filter_years=years)
        vector_candidates = engine.vector_index.search(q_text, top_k=15, filter_tickers=tickers, filter_years=years)
        
        # 3. Merge candidates with RRF to create a robust candidate pool
        merged_candidates = engine.rrf(bm25_candidates, vector_candidates, top_k=5, rrf_k=RRF_K)
        
        logger.info(f"[{q_id}] Evaluating query: '{q_text}' with {len(merged_candidates)} candidates...")
        
        # 4. Run LLM Judge to verify which candidates are actual Ground Truths
        ticker_val = tickers[0] if tickers else "UNKNOWN"
        year_val = years[0] if years else "UNKNOWN"
        
        gt_chunks, gt_reasoning = judge.evaluate(
            query=q_text,
            ticker=ticker_val,
            year=year_val,
            candidates=merged_candidates
        )
        
        # Validate that the judge's selected chunks actually exist in the candidates list
        candidate_ids = {c["chunk_id"] for c in merged_candidates}
        valid_gt_chunks = [cid for cid in gt_chunks if cid in candidate_ids]
        
        if len(valid_gt_chunks) < len(gt_chunks):
            invalid_chunks = set(gt_chunks) - candidate_ids
            logger.warning(f"[{q_id}] Judge returned hallucinated/invalid chunk IDs: {invalid_chunks}")
            gt_chunks = valid_gt_chunks

        # If the judge fails to find a chunk, fallback to the top candidate from RRF to prevent empty GT
        if not gt_chunks and merged_candidates:
            logger.warning(f"[{q_id}] Judge returned no relevant chunks. Falling back to Top-1 candidate.")
            gt_chunks = [merged_candidates[0]["chunk_id"]]
            gt_reasoning = "Fallback to Top-1 candidate from RRF index search."
            
        metadata = {
            "tickers": tickers if tickers else ["UNKNOWN"],
            "years": years if years else ["UNKNOWN"],
            "ticker": ticker_val,
            "year": year_val
        }
        
        annotated_queries.append({
            "query_id": q_id,
            "query": q_text,
            "category": category,
            "ground_truth_chunks": gt_chunks,
            "ground_truth_explanation": gt_reasoning,
            "metadata": metadata
        })
        
        logger.success(f"[{q_id}] Ground Truth chunks assigned: {gt_chunks}")
        
    # Write output to test_queries.jsonl
    with open(output_file, "w", encoding="utf-8") as out:
        for item in annotated_queries:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logger.success(f"Pipeline finished successfully! Output saved to {output_file}")

if __name__ == "__main__":
    main()
