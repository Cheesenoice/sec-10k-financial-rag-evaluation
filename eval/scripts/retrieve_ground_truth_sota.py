"""
eval/scripts/retrieve_ground_truth_sota.py
Ablation Ground Truth Annotation pipeline using SOTA MTEB Models and LLM-as-a-Judge.
Optimized with GTE-Qwen2-1.5B-instruct, ColBERTv2 MaxSim, and batched Gemini 3.5 Flash.
"""

import sys
import os
import json
import re
import time
from pathlib import Path
from loguru import logger
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from google import genai
from google.genai import types

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
from src.config import TARGET_YEARS, GEMINI_API_KEY, GEMINI_API_KEYS, GEMINI_MODEL, GEMINI_REFERER

# Configure English-only logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# =====================================================================
# 1. LLM-AS-A-JUDGE CONFIGURATION (GEMINI-3.5-FLASH VIA GOOGLE GENAI)
# =====================================================================
BATCH_JUDGE_SYSTEM_PROMPT = """You are an expert financial auditor and database annotator.
Your job is to inspect a list of queries and their respective candidate document chunks retrieved from SEC 10-K filings, and decide which candidate chunks contain the exact numerical figures or qualitative facts required to answer each query.

For each query in the list, respond strictly in JSON format.

JSON schema:
{
  "queries": [
    {
      "query_id": "q_01",
      "relevant_chunk_ids": ["CHUNK_ID_1", "CHUNK_ID_2"],
      "reasoning": "Brief explanation of why these chunks are relevant and contain the specific answers."
    },
    ...
  ]
}

If no candidate chunk contains the answer for a query, return an empty list for "relevant_chunk_ids":
{
  "query_id": "q_02",
  "relevant_chunk_ids": [],
  "reasoning": "None of the candidates contain the exact figures or facts for the query."
}
"""

class LLMJudge:
    def __init__(self, api_key: str):
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError("Invalid GEMINI_API_KEY. Gemini API Key is required for the LLM Judge.")
        self.client = genai.Client(
            api_key=api_key,
            http_options={"headers": {"Referer": GEMINI_REFERER}} if GEMINI_REFERER else None
        )
        self.model = GEMINI_MODEL
        
    def evaluate_batch(self, batch_items: list) -> dict:
        """
        batch_items is a list of dicts: {"query_id": q_id, "query": q_text, "ticker": t, "year": y, "candidates": list}
        Returns a dict mapping query_id -> (relevant_chunk_ids, reasoning)
        """
        user_prompt = "Please evaluate the following queries and their candidate chunks:\n"
        for item in batch_items:
            user_prompt += f"\n--- Query ID: {item['query_id']} ---\n"
            user_prompt += f"Query: {item['query']}\n"
            user_prompt += f"Target Company: {item['ticker']}\n"
            user_prompt += f"Target Year: {item['year']}\n"
            user_prompt += "Candidates:\n"
            for i, c in enumerate(item["candidates"]):
                user_prompt += f"  [{i+1}] Chunk ID: {c['chunk_id']}\n  Text: {c['text']}\n"
                
        max_retries = 5
        base_delay = 5.0  # seconds
        
        for attempt in range(max_retries):
            try:
                time.sleep(2.0)
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=BATCH_JUDGE_SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json"
                    )
                )
                
                result = json.loads(response.text.strip())
                queries_res = result.get("queries", [])
                
                output_map = {}
                for q_res in queries_res:
                    q_id = q_res.get("query_id")
                    gt_chunks = q_res.get("relevant_chunk_ids", [])
                    reasoning = q_res.get("reasoning", "No reasoning provided.")
                    if q_id:
                        output_map[q_id] = (gt_chunks, reasoning)
                return output_map
            except Exception as e:
                err_str = str(e)
                if any(x in err_str or x in err_str.lower() for x in ["403", "429", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED", "suspended", "blocked", "quota"]):
                    logger.error(f"API quota/rate limit/error: {e}. Raising exception to trigger client rotation...")
                    raise e
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Error calling Gemini API for Judge batch: {e}. Retrying in {delay:.1f}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                
        logger.error(f"Failed to evaluate judge batch after {max_retries} attempts.")
        raise RuntimeError("Failed to evaluate judge batch after all retries.")

# =====================================================================
# 2. RUNNER ENGINE WITH INDEPENDENT DENSE INDEX & COLBERT-MAXSIM
# =====================================================================
class AnnotationEngine:
    def __init__(self):
        logger.info("Initializing search indexes for candidate generation...")
        self.bm25_index = BM25Index()
        self.bm25_index.load()
        
        # Load corpus
        self.load_corpus()
        
        # Load GTE-Qwen2 embeddings from cache if available
        cache_path = ROOT_DIR / "data/indexes/gte_qwen_embeddings.npy"
        if cache_path.exists():
            logger.info("Loading GTE-Qwen2 embeddings from cache...")
            self.chunk_embeddings = np.load(cache_path)
        else:
            self.chunk_embeddings = None
            
        self.dense_model = None
        self.colbert_tokenizer = None
        self.colbert_model = None
        self.colbert_doc_cache = {}

    def load_corpus(self):
        docs_file = ROOT_DIR / "data/processed/documents.jsonl"
        self.docs = []
        with open(docs_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.docs.append(json.loads(line))
        logger.info(f"Loaded {len(self.docs)} chunks for SOTA corpus.")

    def init_gte_qwen(self):
        if self.dense_model is not None:
            return
            
        logger.info("Initializing GTE-Qwen2-1.5B-instruct...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        try:
            self.dense_model = SentenceTransformer(
                "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                trust_remote_code=True,
                local_files_only=True,
                model_kwargs={"torch_dtype": dtype}
            ).to(device)
        except Exception:
            self.dense_model = SentenceTransformer(
                "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                trust_remote_code=True,
                model_kwargs={"torch_dtype": dtype}
            ).to(device)
        
        cache_path = ROOT_DIR / "data/indexes/gte_qwen_embeddings.npy"
        if cache_path.exists():
            logger.info("Loading GTE-Qwen2 embeddings from cache...")
            self.chunk_embeddings = np.load(cache_path)
        else:
            logger.info("Generating GTE-Qwen2 embeddings for all chunks...")
            corpus_texts = [d["text"] for d in self.docs]
            embeddings = self.dense_model.encode(
                corpus_texts,
                batch_size=16,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            logger.info(f"Saving GTE-Qwen2 embeddings to cache at {cache_path}...")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            self.chunk_embeddings = embeddings

    def encode_queries(self, queries: list) -> np.ndarray:
        logger.info("Loading GTE-Qwen2-1.5B-instruct to encode queries...")
        self.init_gte_qwen()
        
        instruct_queries = [
            f"Instruct: Given a financial query, retrieve relevant passages that answer the query.\nQuery: {q}"
            for q in queries
        ]
        
        logger.info(f"Encoding {len(queries)} queries...")
        q_vectors = self.dense_model.encode(
            instruct_queries,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Unload GTE-Qwen2 model to free VRAM
        logger.info("Unloading GTE-Qwen2-1.5B-instruct to free GPU memory...")
        del self.dense_model
        self.dense_model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return q_vectors

    def init_colbert(self):
        if self.colbert_model is not None:
            return
        logger.info("Initializing ColBERTv2...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.colbert_tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0", local_files_only=True)
            self.colbert_model = AutoModel.from_pretrained("colbert-ir/colbertv2.0", local_files_only=True).to(device)
        except Exception:
            self.colbert_tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
            self.colbert_model = AutoModel.from_pretrained("colbert-ir/colbertv2.0").to(device)

    def dense_search(self, query: str, top_k: int = 30, filter_tickers: list = None, filter_years: list = None) -> list:
        # Prepend instruct prefix for query
        instruct_query = f"Instruct: Given a financial query, retrieve relevant passages that answer the query.\nQuery: {query}"
        
        self.init_gte_qwen()
        q_vector = self.dense_model.encode([instruct_query], convert_to_numpy=True)
        
        # Calculate cosine similarities
        q_vector_norm = q_vector / np.linalg.norm(q_vector, axis=1, keepdims=True)
        chunk_embs_norm = self.chunk_embeddings / np.linalg.norm(self.chunk_embeddings, axis=1, keepdims=True)
        
        scores = np.dot(chunk_embs_norm, q_vector_norm.T).squeeze()
        
        results = []
        for idx, score in enumerate(scores):
            doc = self.docs[idx]
            meta = doc["metadata"]
            if filter_tickers and meta["ticker"] not in filter_tickers:
                continue
            if filter_years and meta["year"] not in filter_years:
                continue
            results.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "score": float(score),
                "metadata": meta
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def dense_search_with_vector(self, q_vector: np.ndarray, top_k: int = 30, filter_tickers: list = None, filter_years: list = None) -> list:
        if q_vector.ndim == 1:
            q_vector = np.expand_dims(q_vector, axis=0)
            
        q_vector_norm = q_vector / np.linalg.norm(q_vector, axis=1, keepdims=True)
        chunk_embs_norm = self.chunk_embeddings / np.linalg.norm(self.chunk_embeddings, axis=1, keepdims=True)
        
        scores = np.dot(chunk_embs_norm, q_vector_norm.T).squeeze()
        
        results = []
        for idx, score in enumerate(scores):
            doc = self.docs[idx]
            meta = doc["metadata"]
            if filter_tickers and meta["ticker"] not in filter_tickers:
                continue
            if filter_years and meta["year"] not in filter_years:
                continue
            results.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "score": float(score),
                "metadata": meta
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def compute_colbert_scores(self, query: str, candidates: list) -> list:
        if not candidates:
            return []
        
        self.init_colbert()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Identify which candidates need to be encoded
        uncached_candidates = []
        for c in candidates:
            cid = c["chunk_id"]
            if cid not in self.colbert_doc_cache:
                uncached_candidates.append(c)
                
        # 2. Encode uncached candidates
        if uncached_candidates:
            doc_texts = [c["text"] for c in uncached_candidates]
            d_tokens = self.colbert_tokenizer(doc_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            
            with torch.no_grad():
                d_out = self.colbert_model(**d_tokens).last_hidden_state
                if hasattr(self.colbert_model, 'linear'):
                    d_out = self.colbert_model.linear(d_out)
                d_emb_all = torch.nn.functional.normalize(d_out, p=2, dim=-1)
                d_mask_all = d_tokens["attention_mask"].bool()
                
            # Store in CPU cache to preserve GPU memory
            for idx, c in enumerate(uncached_candidates):
                cid = c["chunk_id"]
                self.colbert_doc_cache[cid] = (
                    d_emb_all[idx].cpu(),
                    d_mask_all[idx].cpu()
                )
                
        # 3. Retrieve embeddings for all candidates and move to GPU
        d_embs_list = []
        d_masks_list = []
        max_doc_len = 0
        
        for c in candidates:
            cid = c["chunk_id"]
            emb, mask = self.colbert_doc_cache[cid]
            d_embs_list.append(emb)
            d_masks_list.append(mask)
            max_doc_len = max(max_doc_len, emb.shape[0])
            
        # Pad to max_doc_len for batching
        padded_embs = []
        padded_masks = []
        
        for emb, mask in zip(d_embs_list, d_masks_list):
            curr_len = emb.shape[0]
            pad_len = max_doc_len - curr_len
            if pad_len > 0:
                padded_emb = torch.cat([emb, torch.zeros(pad_len, emb.shape[1])], dim=0)
                padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.bool)], dim=0)
            else:
                padded_emb = emb
                padded_mask = mask
            padded_embs.append(padded_emb)
            padded_masks.append(padded_mask)
            
        d_emb = torch.stack(padded_embs).to(device)
        d_mask = torch.stack(padded_masks).to(device)
        
        # 4. Encode query
        q_tokens = self.colbert_tokenizer([query], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            q_out = self.colbert_model(**q_tokens).last_hidden_state
            if hasattr(self.colbert_model, 'linear'):
                q_out = self.colbert_model.linear(q_out)
            q_emb = torch.nn.functional.normalize(q_out, p=2, dim=-1)
            
        num_docs = len(candidates)
        q_len = q_emb.shape[1]
        
        q_emb_expanded = q_emb.expand(num_docs, -1, -1)
        q_mask = q_tokens["attention_mask"].bool().expand(num_docs, -1)
        
        # Similarity matrix [num_docs, q_len, d_len]
        sim_matrix = torch.matmul(q_emb_expanded, d_emb.transpose(1, 2))
        sim_matrix = sim_matrix.masked_fill(~d_mask.unsqueeze(1), -1e9)
        
        # MaxSim
        max_sim, _ = torch.max(sim_matrix, dim=2)
        max_sim = max_sim.masked_fill(~q_mask, 0.0)
        
        scores = torch.sum(max_sim, dim=1).cpu().numpy()
        
        for c, score in zip(candidates, scores):
            c["score"] = float(score)
            
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

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
    
    judges = []
    for key in GEMINI_API_KEYS:
        if key and key != "your_gemini_api_key_here":
            try:
                judges.append(LLMJudge(key))
            except Exception as e:
                logger.error(f"Failed to initialize judge: {e}")
    if not judges:
        raise ValueError("No valid GEMINI_API_KEYS available for LLM Judge.")
    
    queries_raw = []
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("{") and line_str.endswith("}"):
                try:
                    queries_raw.append(json.loads(line_str))
                except Exception:
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
    
    # 1. Pre-encode all queries in a single batch
    logger.info("Batch encoding all queries with GTE-Qwen2...")
    q_texts = [item["query"] for item in queries_raw]
    q_vectors = engine.encode_queries(q_texts)
    
    # 2. Retrieve candidates for all queries
    candidate_pools = []
    for idx, item in enumerate(queries_raw):
        q_text = item["query"]
        category = item.get("category", "factual")
        q_id = item.get("query_id", f"q_{idx+1:02d}")
            
        tickers, years = engine.route_query(q_text)
        expanded_q = engine.expand_query(q_text)
        
        # Search using BM25
        bm25_candidates = engine.bm25_index.search(expanded_q, top_k=30, filter_tickers=tickers, filter_years=years)
        
        # Search using GTE-Qwen2-1.5B-instruct precomputed vector
        dense_candidates = engine.dense_search_with_vector(q_vectors[idx], top_k=30, filter_tickers=tickers, filter_years=years)
        
        # Unique union
        unique_candidates = {}
        for c in bm25_candidates + dense_candidates:
            unique_candidates[c["chunk_id"]] = c
            
        cand_list = list(unique_candidates.values())
        
        # Late Interaction Reranking with ColBERTv2 MaxSim
        reranked_candidates = engine.compute_colbert_scores(q_text, cand_list)
        top_candidates = reranked_candidates[:5]
        
        ticker_val = tickers[0] if tickers else "UNKNOWN"
        year_val = years[0] if years else "UNKNOWN"
        
        candidate_pools.append({
            "query_id": q_id,
            "query": q_text,
            "category": category,
            "ticker": ticker_val,
            "year": year_val,
            "tickers": tickers,
            "years": years,
            "candidates": top_candidates,
            "raw_explanation": item.get("ground_truth_explanation", "No explanation provided.")
        })
        
        # Periodic cleanup during long loop
        if idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Free ColBERT model to release GPU memory
    logger.info("Unloading ColBERTv2 to release GPU memory...")
    del engine.colbert_model
    engine.colbert_model = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Run LLM Judge in batches of 10 in parallel
    logger.info("Evaluating candidates with Batched LLM Judge (Gemini 3.5 Flash)...")
    batches = []
    batch_size = 10
    for i in range(0, len(candidate_pools), batch_size):
        batches.append(candidate_pools[i:i+batch_size])

    def process_judge_batch_task(batch_idx, batch_items):
        judge_instance = judges[batch_idx % len(judges)]
        try:
            return judge_instance.evaluate_batch(batch_items)
        except Exception as e:
            logger.error(f"Error in judge batch {batch_idx+1}: {e}")
            for alt_judge in judges:
                if alt_judge == judge_instance:
                    continue
                try:
                    return alt_judge.evaluate_batch(batch_items)
                except Exception:
                    continue
            return {}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = len(judges)
    logger.info(f"Starting ThreadPoolExecutor with {max_workers} workers to run LLM Judge in parallel...")
    
    annotated_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_judge_batch_task, idx, batch): idx for idx, batch in enumerate(batches)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                batch_output = future.result()
                annotated_results.update(batch_output)
                logger.info(f"Judge batch {idx+1}/{len(batches)} finished.")
            except Exception as e:
                logger.error(f"Judge batch {idx+1} raised exception: {e}")
        
    # 3. Format and save test_queries.jsonl
    annotated_queries = []
    for item in candidate_pools:
        q_id = item["query_id"]
        q_text = item["query"]
        category = item["category"]
        candidates = item["candidates"]
        
        gt_chunks, gt_reasoning = annotated_results.get(q_id, ([], None))
        if not gt_reasoning or gt_reasoning == "Failed to retrieve judge evaluation.":
            gt_reasoning = item.get("raw_explanation", "Failed to retrieve judge evaluation.")
        
        # Post-validation and fallback
        candidate_ids = {c["chunk_id"] for c in candidates}
        valid_gt_chunks = [cid for cid in gt_chunks if cid in candidate_ids]
        
        if len(valid_gt_chunks) < len(gt_chunks):
            invalid_chunks = set(gt_chunks) - candidate_ids
            logger.warning(f"[{q_id}] Judge returned invalid/hallucinated chunk IDs: {invalid_chunks}")
            gt_chunks = valid_gt_chunks
            
        if not gt_chunks and candidates:
            logger.warning(f"[{q_id}] Judge returned no relevant chunks. Falling back to Top-1 candidate from ColBERTv2.")
            gt_chunks = [candidates[0]["chunk_id"]]
            gt_reasoning = f"Fallback to Top-1 candidate from ColBERTv2 MaxSim reranking. Original QGen Critic reasoning: {item.get('raw_explanation')}"
            
        metadata = {
            "tickers": item["tickers"] if item["tickers"] else ["UNKNOWN"],
            "years": item["years"] if item["years"] else ["UNKNOWN"],
            "ticker": item["ticker"],
            "year": item["year"]
        }
        
        annotated_queries.append({
            "query_id": q_id,
            "query": q_text,
            "category": category,
            "ground_truth_chunks": gt_chunks,
            "ground_truth_explanation": gt_reasoning,
            "metadata": metadata
        })
        
        logger.success(f"[{q_id}] Assigned GT chunks: {gt_chunks}")
        
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        for item in annotated_queries:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logger.success(f"SOTA Ground Truth annotation finished! Output saved to: {output_file}")

if __name__ == "__main__":
    main()

