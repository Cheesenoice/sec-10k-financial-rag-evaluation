import os
from loguru import logger
from groq import Groq
import ollama
from google import genai
from google.genai import types
from src.config import GROQ_API_KEY, GROQ_MODEL, GEMINI_API_KEY, GEMINI_API_KEYS, GEMINI_MODEL, GEMINI_REFERER

class LLMClient:
    def __init__(self, use_local: bool = False):
        self.use_local = use_local
        self.gemini_clients = []
        self.current_client_idx = 0
        self.groq_client = None
        
        if not use_local:
            # 1. Try Gemini APIs
            for api_key in GEMINI_API_KEYS:
                if api_key and api_key != "your_gemini_api_key_here":
                    try:
                        client = genai.Client(
                            api_key=api_key,
                            http_options={"headers": {"Referer": GEMINI_REFERER}} if GEMINI_REFERER else None
                        )
                        self.gemini_clients.append(client)
                    except Exception as e:
                        logger.error(f"Kết nối Gemini API key {api_key[:8]}... thất bại: {e}")
            
            if self.gemini_clients:
                self.gemini_client = self.gemini_clients[0]
                logger.info(f"Đã kết nối {len(self.gemini_clients)} Google Gemini API clients. Model mặc định: {GEMINI_MODEL}")
            
            # 2. Try Groq API as fallback if Gemini key is not configured or failed
            if not self.gemini_clients:
                if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
                    try:
                        self.groq_client = Groq(api_key=GROQ_API_KEY)
                        logger.info(f"Đã kết nối Groq Cloud API. Model mặc định: {GROQ_MODEL}")
                    except Exception as e:
                        logger.error(f"Kết nối Groq thất bại: {e}. Chuyển sang dùng Ollama local.")
                        self.use_local = True
                else:
                    logger.warning("Cả GEMINI_API_KEY/KEYS và GROQ_API_KEY đều chưa được cấu hình hợp lệ. Chuyển sang Ollama local.")
                    self.use_local = True
                    
        if self.use_local:
            logger.info("Đang khởi chạy LLM qua Ollama local (Yêu cầu ứng dụng Ollama đang chạy).")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sinh câu trả lời từ LLM với cấu hình temperature=0.0 (chống bịa đặt)
        """
        if not self.use_local:
            # Try Gemini if client is active
            if self.gemini_clients:
                # Round-robin client selection
                client = self.gemini_clients[self.current_client_idx]
                self.current_client_idx = (self.current_client_idx + 1) % len(self.gemini_clients)
                
                try:
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.0,
                        )
                    )
                    return response.text.strip()
                except Exception as e:
                    logger.error(f"Lỗi gọi Gemini API (idx={self.current_client_idx-1}): {e}. Tiến hành fallback sang key khác...")
                    # Fallback rotation over all other clients
                    for idx, alt_client in enumerate(self.gemini_clients):
                        if alt_client == client:
                            continue
                        try:
                            response = alt_client.models.generate_content(
                                model=GEMINI_MODEL,
                                contents=user_prompt,
                                config=types.GenerateContentConfig(
                                    system_instruction=system_prompt,
                                    temperature=0.0,
                                )
                            )
                            return response.text.strip()
                        except Exception:
                            continue
                    logger.error("Tất cả Gemini clients đều lỗi. Chuyển sang Groq/Ollama...")
            
            # Try Groq if client is active
            if self.groq_client:
                try:
                    chat_completion = self.groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=GROQ_MODEL,
                        temperature=0.0,
                        max_tokens=1024
                    )
                    return chat_completion.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"Lỗi gọi Groq API: {e}. Tiến hành fallback sang Ollama...")
                    self.use_local = True
                
        # Gọi Ollama local
        try:
            response = ollama.chat(
                model='llama3.2',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={'temperature': 0.0}
            )
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"Lỗi gọi Ollama local: {e}. Vui lòng kiểm tra ứng dụng Ollama đã bật chưa.")
            return "Error: Không thể kết nối tới bất kỳ LLM nào (Gemini/Groq Cloud lỗi và Ollama Local chưa bật)."
