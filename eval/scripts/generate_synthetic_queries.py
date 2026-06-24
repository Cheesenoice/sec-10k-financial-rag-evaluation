"""
eval/scripts/generate_synthetic_queries.py
Actual Synthetic Query Generation (QGen) pipeline.
Loads SEC 10-K document chunks, selects high-value financial passages,
generates target-focused questions, and validates them against rules.
Optimized with Gemini 3.1 Flash Lite batching.
"""

import os
import sys
import json
import time
from pathlib import Path
from loguru import logger
from google import genai
from google.genai import types

# Reconfigure stdout/stderr for Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import GEMINI_API_KEYS, GEMINI_MODEL_LITE, TARGET_TICKERS, TARGET_YEARS, GEMINI_REFERER

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# =====================================================================
# 1. GENERATION PROMPTS
# =====================================================================

BATCH_SYSTEM_PROMPT = """You are a Financial NLP QA Dataset Engineer.
Your task is to generate high-quality, realistic financial questions based on the provided list of SEC 10-K document chunks.

For each item in the list, generate one question matching the requested category:
1. factual: Ask for a direct financial figure or fact explicitly present in the chunk. The answer must be a single number or fact.
2. comparison: Compare a financial metric across different years/segments or companies. If multiple document chunks are provided (e.g. Document Chunk 1 and Document Chunk 2), formulate a query that compares metrics across both chunks (e.g., comparing net income of Apple in Chunk 1 with Microsoft in Chunk 2, or comparing Apple's R&D spend in 2023 with 2022). If only one chunk is provided, compare metrics or segments within that single chunk.
3. lexical_gap: Identify a major financial metric mentioned in the text (e.g. "capital expenditures", "research and development", "net income") and replace it in your question with a standard industry synonym or abbreviation (e.g., "capex", "R&D spend", "net earnings"). NEVER use a synonym for a metric that is NOT present in the text (e.g., do not ask about "capex" if the text only discusses tax benefits).
4. temporal_routing: Formulate the query to contain a specific fiscal year (e.g., "in fiscal year 2024", "for 2023") that is explicitly present in the chunk.

Rules:
- STRICT GROUNDING: Every noun, metric, and year in the question must be directly supported by the text. Do NOT hallucinate or combine unrelated concepts.
- Always explicitly mention the target company name (e.g., Apple, Microsoft, Nvidia, Tesla, Amazon, Alphabet/Google). If multiple chunks from different companies are provided, mention both.
- Respond ONLY as a valid JSON array of objects.

JSON schema:
[
  {
    "index": 0,
    "query": "Generated question"
  },
  ...
]
"""

CRITIC_SYSTEM_PROMPT = """You are a Quality Assurance critic.
For each item in the list (containing a source chunk and a generated question), evaluate if the question is fully answerable from the chunk and does not contain hallucinated facts.

CRITICAL: Standard financial synonyms and abbreviations (e.g. 'capex' for 'capital expenditures', 'R&D spend' or 'R&D expenses' for 'research and development expense', 'net earnings' or 'net income' for 'net profit', 'sales' or 'revenues' for 'net sales') are EXPLICITLY ALLOWED. Do NOT mark them as hallucinations or unanswerable. Hallucination applies ONLY if the query asks for numerical figures, years, or companies completely absent from the source chunk.

Respond ONLY as a valid JSON array of objects.

JSON schema:
[
  {
    "index": 0,
    "answerability": 1-5,
    "hallucination": true/false,
    "reasoning": "Brief explanation of why the question is answerable from the chunk, citing the specific figures or facts."
  },
  ...
]
"""

# =====================================================================
# 2. GENERATION PIPELINE ENGINE
# =====================================================================

class SyntheticQGenPipeline:
    def __init__(self, api_keys: list):
        if not api_keys:
            raise ValueError("GEMINI_API_KEYS is not configured or empty.")
        self.clients = []
        for key in api_keys:
            if key and key != "your_gemini_api_key_here":
                try:
                    client = genai.Client(
                        api_key=key,
                        http_options={"headers": {"Referer": GEMINI_REFERER}} if GEMINI_REFERER else None
                    )
                    self.clients.append(client)
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini Client: {e}")
        if not self.clients:
            raise ValueError("No valid GEMINI_API_KEYS available.")
        self.model = GEMINI_MODEL_LITE

    def select_source_chunks(self, docs_file: Path, limit_per_company: int = 70) -> list:
        """
        Loads document chunks and selects high-value, diverse, and section-balanced paragraphs.
        """
        logger.info(f"Loading document chunks from {docs_file}...")
        chunks = []
        with open(docs_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        
        # 1. Group candidate chunks by ticker and section (Item 7 vs Item 8)
        candidates_by_ticker = {ticker: {"Item 7": [], "Item 8": []} for ticker in TARGET_TICKERS}
        
        financial_keywords = [
            "revenue", "sales", "net income", "operating income", "operating expense",
            "capital expenditures", "capex", "research and development", "r&d",
            "cash and cash equivalents", "total assets", "operating activities"
        ]
        
        for c in chunks:
            ticker = c.get("metadata", {}).get("ticker")
            year = c.get("metadata", {}).get("year")
            section = c.get("metadata", {}).get("section")
            text = c.get("text", "")
            
            # Target checks
            if ticker not in TARGET_TICKERS or year not in TARGET_YEARS:
                continue
            if section not in ["Item 7", "Item 8"]:
                continue
            if len(text) < 200:
                continue
                
            # Filter clean text (ignore raw html boilerplate)
            html_noise = text.count("<") + text.count(">") + text.count("&nbsp;")
            if html_noise > len(text) * 0.05:
                continue
                
            # Calculate Financial Density Score (FDS)
            text_lower = text.lower()
            keyword_score = sum(text_lower.count(kw) for kw in financial_keywords)
            symbol_count = text.count("$") + text.count("%")
            unit_count = text_lower.count("million") + text_lower.count("billion")
            entity_score = symbol_count + unit_count
            
            num_digits = sum(char.isdigit() for char in text)
            digit_ratio = num_digits / len(text)
            
            # FDS Formula
            fds_score = (keyword_score * 1.5) + (entity_score * 2.0) + (digit_ratio * 30.0) - (html_noise * 0.5)
            fds_score = max(fds_score, 0.0)
            
            # Must have at least some relevance or numeric info
            if fds_score < 1.0:
                continue
                
            candidates_by_ticker[ticker][section].append({
                "chunk": c,
                "score": fds_score
            })
            
        logger.info("Initializing BAAI/bge-small-en-v1.5 to perform MMR semantic diversity filtering...")
        embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        selected_chunks = []
        target_per_section = limit_per_company // 2
        
        for ticker in TARGET_TICKERS:
            for section in ["Item 7", "Item 8"]:
                section_candidates = candidates_by_ticker[ticker][section]
                if not section_candidates:
                    logger.warning(f"No candidates found for {ticker} {section}")
                    continue
                
                # Sort initially by score
                section_candidates.sort(key=lambda x: x["score"], reverse=True)
                
                # Limit candidates to pool of top 100 to keep MMR fast
                pool = section_candidates[:100]
                texts = [c["chunk"]["text"] for c in pool]
                scores = [c["score"] for c in pool]
                
                # Encode embeddings
                embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
                
                # Normalize embeddings
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                norm_embs = embeddings / norms
                
                # MMR Selection Loop
                selected_indices = []
                unselected_indices = list(range(len(pool)))
                K = min(target_per_section, len(pool))
                
                max_score = max(scores) if scores else 1.0
                if max_score == 0:
                    max_score = 1.0
                norm_scores = [s / max_score for s in scores]
                
                for step in range(K):
                    if not unselected_indices:
                        break
                    if step == 0:
                        best_idx = max(unselected_indices, key=lambda i: norm_scores[i])
                    else:
                        best_idx = None
                        best_mmr = -float("inf")
                        selected_embs = norm_embs[selected_indices]
                        
                        for i in unselected_indices:
                            sims = np.dot(selected_embs, norm_embs[i])
                            max_sim = np.max(sims)
                            
                            # MMR score: lambda=0.5
                            mmr_val = 0.5 * norm_scores[i] - 0.5 * max_sim
                            if mmr_val > best_mmr:
                                best_mmr = mmr_val
                                best_idx = i
                                
                    if best_idx is not None:
                        selected_indices.append(best_idx)
                        unselected_indices.remove(best_idx)
                        
                for idx in selected_indices:
                    selected_chunks.append(pool[idx]["chunk"])
                    
        # Check balance
        item7_count = sum(1 for c in selected_chunks if c["metadata"]["section"] == "Item 7")
        item8_count = sum(1 for c in selected_chunks if c["metadata"]["section"] == "Item 8")
        
        logger.info(f"Selected {len(selected_chunks)} source chunks using FDS & MMR.")
        logger.info(f"Section Balance: Item 7 (MD&A) = {item7_count} chunks, Item 8 (Financials) = {item8_count} chunks.")
        
        return selected_chunks

    def find_pairing_chunk(self, primary_chunk: dict, all_chunks: list) -> dict:
        p_meta = primary_chunk.get("metadata", {})
        p_ticker = p_meta.get("ticker")
        p_year = p_meta.get("year")
        
        # Try to find a chunk of the same ticker but different year first (cross-year)
        for c in all_chunks:
            c_meta = c.get("metadata", {})
            if c_meta.get("ticker") == p_ticker and c_meta.get("year") != p_year:
                return c
                
        # Try to find a chunk of a different ticker but same year (cross-company)
        for c in all_chunks:
            c_meta = c.get("metadata", {})
            if c_meta.get("ticker") != p_ticker and c_meta.get("year") == p_year:
                return c
                
        return None

    def generate_queries_batch(self, batch_items: list, start_client_idx: int) -> dict:
        """
        Generates queries for a batch of chunks.
        batch_items is a list of dicts: {"index": i, "chunk": chunk, "chunk2": chunk2, "category": cat}
        """
        user_prompt = "Generate questions for the following items:\n"
        for item in batch_items:
            chunk = item["chunk"]
            chunk2 = item.get("chunk2")
            user_prompt += f"\n--- Item {item['index']} ---\n"
            user_prompt += f"Category: {item['category']}\n"
            
            if chunk2:
                user_prompt += f"Target Company 1: {chunk.get('metadata', {}).get('ticker')}\n"
                user_prompt += f"Target Year 1: {chunk.get('metadata', {}).get('year')}\n"
                user_prompt += f"Target Company 2: {chunk2.get('metadata', {}).get('ticker')}\n"
                user_prompt += f"Target Year 2: {chunk2.get('metadata', {}).get('year')}\n"
                user_prompt += f"Document Chunk 1:\n\"\"\"\n{chunk['text']}\n\"\"\"\n"
                user_prompt += f"Document Chunk 2:\n\"\"\"\n{chunk2['text']}\n\"\"\"\n"
            else:
                user_prompt += f"Target Company: {chunk.get('metadata', {}).get('ticker')}\n"
                user_prompt += f"Target Fiscal Year: {chunk.get('metadata', {}).get('year')}\n"
                user_prompt += f"Section: {chunk.get('metadata', {}).get('section')}\n"
                user_prompt += f"Document Chunk:\n\"\"\"\n{chunk['text']}\n\"\"\"\n"

        num_clients = len(self.clients)
        for attempt in range(num_clients):
            client_idx = (start_client_idx + attempt) % num_clients
            client = self.clients[client_idx]
            try:
                # Add slight delay to not trigger concurrent QPS limits on same key
                time.sleep(1.0)
                response = client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=BATCH_SYSTEM_PROMPT,
                        temperature=0.7,
                        response_mime_type="application/json"
                    )
                )
                
                results = json.loads(response.text.strip())
                # Map queries back by index
                queries_map = {}
                for res in results:
                    idx = res.get("index")
                    q_text = res.get("query", "").replace('"', '').replace("'", "").strip()
                    if idx is not None:
                        queries_map[idx] = q_text
                return queries_map
            except Exception as e:
                logger.warning(f"Failed to generate queries batch using client {client_idx}: {e}. Retrying with next client...")
        logger.error("All clients failed to generate queries for this batch.")
        return {}

    def validate_query(self, query: str, chunk: dict, category: str) -> bool:
        """
        Verifies query quality:
        """
        if not query or len(query.split()) < 5:
            return False
            
        ticker = chunk.get("metadata", {}).get("ticker", "").lower()
        query_lower = query.lower()
        
        # Verify company mention
        ticker_mapping = {
            "aapl": ["apple"],
            "msft": ["microsoft"],
            "tsla": ["tesla"],
            "nvda": ["nvidia"],
            "amzn": ["amazon"],
            "googl": ["google", "alphabet"]
        }
        
        allowed_names = [ticker] + ticker_mapping.get(ticker, [])
        if not any(name in query_lower for name in allowed_names if name):
            logger.warning(f"Validation Rejected: Query does not mention company name: '{query}'")
            return False
            
        # Verify year presence for temporal routing
        if category == "temporal_routing":
            year = str(chunk.get("metadata", {}).get("year", ""))
            if year not in query_lower:
                logger.warning(f"Validation Rejected: Temporal query missing year {year}: '{query}'")
                return False
                
        return True

    def validate_queries_batch(self, batch_items: list, start_client_idx: int) -> dict:
        """
        Evaluates a batch of generated queries using LLM Critic.
        batch_items is a list of dicts: {"index": i, "chunk": chunk, "chunk2": chunk2, "query": query}
        Returns a dict mapping index -> validation boolean.
        """
        user_prompt = "Evaluate the following generated questions:\n"
        for item in batch_items:
            chunk = item["chunk"]
            chunk2 = item.get("chunk2")
            user_prompt += f"\n--- Item {item['index']} ---\n"
            if chunk2:
                user_prompt += f"Source Chunk 1:\n\"\"\"\n{chunk['text']}\n\"\"\"\n"
                user_prompt += f"Source Chunk 2:\n\"\"\"\n{chunk2['text']}\n\"\"\"\n"
            else:
                user_prompt += f"Source Chunk:\n\"\"\"\n{chunk['text']}\n\"\"\"\n"
            user_prompt += f"Question: \"{item['query']}\"\n"

        num_clients = len(self.clients)
        for attempt in range(num_clients):
            client_idx = (start_client_idx + attempt) % num_clients
            client = self.clients[client_idx]
            try:
                time.sleep(1.0)
                response = client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=CRITIC_SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json"
                    )
                )
                
                results = json.loads(response.text.strip())
                validation_map = {}
                for res in results:
                    idx = res.get("index")
                    answerability = res.get("answerability", 1)
                    hallucination = res.get("hallucination", True)
                    reasoning = res.get("reasoning", "No explanation provided.")
                    
                    if idx is not None:
                        is_valid = (answerability >= 4 and not hallucination)
                        validation_map[idx] = {"is_valid": is_valid, "reasoning": reasoning}
                        if not is_valid:
                            logger.warning(f"Critic Rejected Index {idx} | Answerability: {answerability}, Hallucination: {hallucination}")
                return validation_map
            except Exception as e:
                logger.warning(f"Failed to validate queries batch using client {client_idx}: {e}. Retrying with next client...")
        logger.error("All clients failed to validate queries for this batch. Falling back to accepting them.")
        return {item["index"]: {"is_valid": True, "reasoning": "Fallback due to API error."} for item in batch_items}

    def run(self, docs_file: Path, output_file: Path, num_queries_to_generate: int = 80):
        selected_chunks = self.select_source_chunks(docs_file)
        
        categories = ["factual", "comparison", "lexical_gap", "temporal_routing"]
        target_per_category = num_queries_to_generate // len(categories)
        
        # Prepare a balanced list of tasks
        all_batch_items = []
        temp_category_counts = {cat: 0 for cat in categories}
        
        chunk_idx = 0
        total_chunks = len(selected_chunks)
        
        while chunk_idx < total_chunks:
            chunk = selected_chunks[chunk_idx]
            chunk_idx += 1
            
            # Select category with lowest count or round robin
            available_categories = [cat for cat in categories if temp_category_counts[cat] < target_per_category]
            if not available_categories:
                category = sorted(temp_category_counts.items(), key=lambda x: x[1])[0][0]
            else:
                category = available_categories[0]
                
            chunk2 = None
            if category == "comparison":
                chunk2 = self.find_pairing_chunk(chunk, selected_chunks)
                
            all_batch_items.append({
                "chunk": chunk,
                "chunk2": chunk2,
                "category": category
            })
            temp_category_counts[category] += 1
            
        batches = []
        batch_size = 10
        for i in range(0, len(all_batch_items), batch_size):
            b_items = all_batch_items[i:i+batch_size]
            for idx, item in enumerate(b_items):
                item["index"] = idx
            batches.append(b_items)
            
        def process_batch_task(batch_idx, batch_items):
            client_idx = batch_idx % len(self.clients)
            queries_map = self.generate_queries_batch(batch_items, client_idx)
            
            stage1_passed = []
            for item in batch_items:
                idx = item["index"]
                query = queries_map.get(idx, "")
                item["query"] = query
                
                if self.validate_query(query, item["chunk"], item["category"]):
                    stage1_passed.append(item)
                else:
                    logger.warning(f"Batch {batch_idx+1} item {idx} failed heuristic validation.")
            
            if not stage1_passed:
                return []
                
            critic_map = self.validate_queries_batch(stage1_passed, client_idx)
            
            batch_results = []
            for item in stage1_passed:
                idx = item["index"]
                query = item["query"]
                chunk = item["chunk"]
                chunk2 = item.get("chunk2")
                category = item["category"]
                
                critic_data = critic_map.get(idx, {"is_valid": False, "reasoning": "No explanation provided."})
                if critic_data.get("is_valid", False):
                    gt_chunks = [chunk["chunk_id"]]
                    if chunk2:
                        gt_chunks.append(chunk2["chunk_id"])
                        
                    record = {
                        "query": query,
                        "category": category,
                        "ground_truth_chunks": gt_chunks,
                        "ground_truth_explanation": critic_data.get("reasoning", "No explanation provided."),
                        "metadata": {
                            "ticker": chunk.get("metadata", {}).get("ticker"),
                            "year": chunk.get("metadata", {}).get("year"),
                            "section": chunk.get("metadata", {}).get("section")
                        }
                    }
                    batch_results.append(record)
            return batch_results

        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = len(self.clients)
        logger.info(f"Starting ThreadPoolExecutor with {max_workers} workers to process {len(batches)} batches in parallel...")
        
        all_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_batch_task, idx, batch): idx for idx, batch in enumerate(batches)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    batch_res = future.result()
                    all_results.extend(batch_res)
                    logger.info(f"Batch {idx+1}/{len(batches)} completed. Obtained {len(batch_res)} valid queries.")
                except Exception as e:
                    logger.error(f"Batch {idx+1} raised exception: {e}")
                    
        # Group results by category
        by_category = {cat: [] for cat in categories}
        for res in all_results:
            by_category[res["category"]].append(res)
            
        # Select round-robin
        final_results = []
        cat_indices = {cat: 0 for cat in categories}
        
        while len(final_results) < num_queries_to_generate:
            added = False
            for cat in categories:
                if len(final_results) >= num_queries_to_generate:
                    break
                idx = cat_indices[cat]
                if idx < len(by_category[cat]):
                    record = by_category[cat][idx]
                    record["query_id"] = f"sq_{len(final_results)+1:02d}"
                    final_results.append(record)
                    cat_indices[cat] += 1
                    added = True
            if not added:
                break
                
        # Save output
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for r in final_results:
                f.write(json.dumps(r) + "\n")
                
        logger.success(f"Successfully selected {len(final_results)} balanced synthetic queries saved to: {output_file}")
        for cat in categories:
            logger.info(f"  Category '{cat}': {sum(1 for r in final_results if r['category'] == cat)} queries")


def main():
    docs_file = ROOT_DIR / "data/processed/documents.jsonl"
    output_file = ROOT_DIR / "data/eval/synthetic_queries_pipeline.jsonl"
    
    if not docs_file.exists():
        logger.error(f"Processed documents not found at {docs_file}. Please run data extraction first.")
        return
        
    try:
        pipeline = SyntheticQGenPipeline(GEMINI_API_KEYS)
        pipeline.run(docs_file, output_file, num_queries_to_generate=300)
    except Exception as e:
        logger.critical(f"Pipeline execution aborted: {e}")

if __name__ == "__main__":
    main()

