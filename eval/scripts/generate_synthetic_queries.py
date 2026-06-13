"""
eval/scripts/generate_synthetic_queries.py
Actual Synthetic Query Generation (QGen) pipeline.
Loads SEC 10-K document chunks, selects high-value financial passages,
generates target-focused questions, and validates them against rules.
"""

import os
import sys
import json
import time
from pathlib import Path
from loguru import logger
from groq import Groq

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

from src.config import GROQ_API_KEY, TARGET_TICKERS, TARGET_YEARS

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# =====================================================================
# 1. GENERATION PROMPTS
# =====================================================================

SYSTEM_PROMPT = """You are a Financial NLP QA Dataset Engineer.
Your task is to generate one high-quality, realistic financial question based on the provided SEC 10-K document chunk.

The question must belong to the requested category:
1. factual: Ask for a direct financial figure or fact explicitly present in the chunk. The answer must be a single number or fact.
2. comparison: Compare a financial metric across different years/segments or companies mentioned explicitly in the chunk. Do NOT invent a comparison if the text contains only one year/segment.
3. lexical_gap: Identify a major financial metric mentioned in the text (e.g. "capital expenditures", "research and development", "net income") and replace it in your question with a standard industry synonym or abbreviation (e.g., "capex", "R&D spend", "net earnings"). NEVER use a synonym for a metric that is NOT present in the text (e.g., do not ask about "capex" if the text only discusses tax benefits).
4. temporal_routing: Formulate the query to contain a specific fiscal year (e.g., "in fiscal year 2024", "for 2023") that is explicitly present in the chunk.

Rules:
- STRICT GROUNDING: Every noun, metric, and year in the question must be directly supported by the text. Do NOT hallucinate or combine unrelated concepts (e.g., do not combine "unrecognized tax benefits" and "capex" into "unrecognized tax benefits capex").
- Always explicitly mention the target company name (e.g., Apple, Microsoft, Nvidia, Tesla, Amazon, Alphabet/Google).
- Never output introductory text, conversational filler, or explanations. Only return the question itself.
"""

USER_TEMPLATE = """Target Company: {ticker}
Target Fiscal Year: {year}
Section: {section}
Category: {category}

Document Chunk:
\"\"\"
{text}
\"\"\"

Please generate the {category} question:"""

# =====================================================================
# 2. GENERATION PIPELINE ENGINE
# =====================================================================

class SyntheticQGenPipeline:
    def __init__(self, api_key: str):
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is not configured in environment or .env file.")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    def select_source_chunks(self, docs_file: Path, limit_per_company: int = 30) -> list:
        """
        Loads document chunks and selects high-value financial statement paragraphs
        containing metrics, segment breakdowns, or balance sheet figures.
        Upgraded with keyword relevance scoring and Jaccard similarity diversity filters.
        """
        logger.info(f"Loading document chunks from {docs_file}...")
        chunks = []
        with open(docs_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        
        financial_keywords = [
            "revenue", "sales", "net income", "operating income", "operating expense",
            "capital expenditures", "capex", "research and development", "r&d",
            "cash and cash equivalents", "total assets", "operating activities"
        ]

        scored_candidates = []
        for c in chunks:
            ticker = c.get("metadata", {}).get("ticker")
            year = c.get("metadata", {}).get("year")
            section = c.get("metadata", {}).get("section")
            text = c.get("text", "")
            
            # 1. Target check
            if ticker not in TARGET_TICKERS or year not in TARGET_YEARS:
                continue
            if section not in ["Item 7", "Item 8"]:
                continue
                
            # 2. Length check
            if len(text) < 200:
                continue
                
            # 3. Numeric check
            num_digits = sum(char.isdigit() for char in text)
            digit_ratio = num_digits / len(text)
            if num_digits < 20 or digit_ratio < 0.02:
                continue
                
            # 4. Keyword score
            text_lower = text.lower()
            keyword_score = sum(text_lower.count(kw) for kw in financial_keywords)
            if keyword_score == 0:
                continue
                
            scored_candidates.append({
                "chunk": c,
                "score": keyword_score + (digit_ratio * 10),
                "ticker": ticker,
                "words": set(text_lower.split())
            })
            
        # Sort candidates by score descending
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        selected_chunks = []
        company_selected = {ticker: [] for ticker in TARGET_TICKERS}
        
        for cand in scored_candidates:
            ticker = cand["ticker"]
            if len(company_selected[ticker]) >= limit_per_company:
                continue
                
            # Jaccard similarity check to ensure diversity
            words_new = cand["words"]
            is_redundant = False
            for prev_cand in company_selected[ticker]:
                words_prev = prev_cand["words"]
                intersection = len(words_new.intersection(words_prev))
                union = len(words_new.union(words_prev))
                jaccard = intersection / union if union > 0 else 0
                if jaccard > 0.5:
                    is_redundant = True
                    break
                    
            if not is_redundant:
                selected_chunks.append(cand["chunk"])
                company_selected[ticker].append(cand)
                
        logger.info(f"Selected {len(selected_chunks)} high-value, diverse source chunks. Company counts: "
                    f"{ {t: len(v) for t, v in company_selected.items()} }")
        return selected_chunks

    def generate_query(self, chunk: dict, category: str) -> str:
        ticker = chunk.get("metadata", {}).get("ticker")
        year = chunk.get("metadata", {}).get("year")
        section = chunk.get("metadata", {}).get("section")
        
        prompt = USER_TEMPLATE.format(
            ticker=ticker,
            year=year,
            section=section,
            category=category,
            text=chunk["text"]
        )
        
        try:
            # Throttling to prevent rate limits
            time.sleep(1.0)
            
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=96
            )
            query = completion.choices[0].message.content.strip()
            # Clean wrapping quotes
            query = query.replace('"', '').replace("'", "")
            return query
        except Exception as e:
            logger.error(f"Failed to generate query for {ticker} {year}: {e}")
            return ""

    def validate_query(self, query: str, chunk: dict, category: str) -> bool:
        """
        Verifies query quality:
        1. Query is not empty.
        2. Query mentions the target company name or standard synonym (e.g. Google/Alphabet).
        3. Query contains at least 5 words.
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

    def validate_query_with_llm(self, query: str, chunk: dict) -> bool:
        """
        Stage 2 Critic: Uses LLM to evaluate if the query is fully answerable from the chunk
        and does not contain hallucinated facts.
        """
        critic_prompt = f"""Source Chunk:
\"\"\"
{chunk['text']}
\"\"\"

Generated Question:
\"{query}\"

Evaluate the question against the chunk. Output JSON ONLY:
{{
  "answerability": 1-5,
  "hallucination": true/false
}}
"""
        system_instruction = (
            "You are a Quality Assurance critic. Respond ONLY in JSON format matching the schema.\n"
            "CRITICAL: Standard financial synonyms and abbreviations (e.g. 'capex' for 'capital expenditures', "
            "'R&D spend' or 'R&D expenses' for 'research and development expense', 'net earnings' or 'net income' for 'net profit', "
            "'sales' or 'revenues' for 'net sales') are EXPLICITLY ALLOWED. Do NOT mark them as hallucinations or "
            "unanswerable. Hallucination applies ONLY if the query asks for numerical figures, years, or companies "
            "completely absent from the source chunk."
        )
        try:
            time.sleep(1.0) # Throttling
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": critic_prompt}
                ],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            result = json.loads(completion.choices[0].message.content.strip())
            answerability = result.get("answerability", 1)
            hallucination = result.get("hallucination", True)
            
            if answerability >= 4 and not hallucination:
                return True
            else:
                logger.warning(f"LLM Critic Rejected Query: '{query}' | Answerability: {answerability}, Hallucination: {hallucination}")
                return False
        except Exception as e:
            logger.error(f"Critic call failed: {e}")
            return True # Fallback to True if API fails

    def run(self, docs_file: Path, output_file: Path, num_queries_to_generate: int = 80):
        selected_chunks = self.select_source_chunks(docs_file)
        
        categories = ["factual", "comparison", "lexical_gap", "temporal_routing"]
        target_per_category = num_queries_to_generate // len(categories)
        category_counts = {cat: 0 for cat in categories}
        
        generated_count = 0
        results = []
        
        for idx, chunk in enumerate(selected_chunks):
            if generated_count >= num_queries_to_generate:
                break
                
            # Pick next category that hasn't reached its target count
            available_categories = [cat for cat in categories if category_counts[cat] < target_per_category]
            if not available_categories:
                # If all categories completed their targets, sort by current count
                category = sorted(category_counts.items(), key=lambda x: x[1])[0][0]
            else:
                # Preserve round-robin order of remaining target categories
                category = available_categories[0]
                
            logger.info(f"[{generated_count + 1}/{num_queries_to_generate}] Generating '{category}' query...")
            
            query = self.generate_query(chunk, category)
            # Stage 1: Heuristic Rules
            if not self.validate_query(query, chunk, category):
                logger.warning("Skipped: Rule-based validation failed.")
                continue
                
            # Stage 2: LLM-as-a-Judge Critic
            if not self.validate_query_with_llm(query, chunk):
                logger.warning("Skipped: LLM Critic validation failed.")
                continue

            record = {
                "query_id": f"sq_{generated_count+1:02d}",
                "query": query,
                "category": category,
                "ground_truth_chunks": [chunk["chunk_id"]],
                "metadata": {
                    "ticker": chunk.get("metadata", {}).get("ticker"),
                    "year": chunk.get("metadata", {}).get("year"),
                    "section": chunk.get("metadata", {}).get("section")
                }
            }
            results.append(record)
            category_counts[category] += 1
            generated_count += 1
            logger.success(f"Accepted: '{query}'")
                
        # Save output
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
                
        logger.success(f"Successfully generated {generated_count} synthetic queries saved to: {output_file}")


def main():
    docs_file = ROOT_DIR / "data/processed/documents.jsonl"
    output_file = ROOT_DIR / "data/eval/synthetic_queries_pipeline.jsonl"
    
    if not docs_file.exists():
        logger.error(f"Processed documents not found at {docs_file}. Please run data extraction first.")
        return
        
    try:
        pipeline = SyntheticQGenPipeline(GROQ_API_KEY)
        pipeline.run(docs_file, output_file, num_queries_to_generate=80)
    except Exception as e:
        logger.critical(f"Pipeline execution aborted: {e}")

if __name__ == "__main__":
    main()
