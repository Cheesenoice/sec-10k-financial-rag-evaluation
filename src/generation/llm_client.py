"""
src/generation/llm_client.py
Đóng gói kết nối LLM (Groq Cloud API & Ollama Local Fallback)
"""

import os
from loguru import logger
from groq import Groq
import ollama
from src.config import GROQ_API_KEY, GROQ_MODEL

class LLMClient:
    def __init__(self, use_local: bool = False):
        self.use_local = use_local
        self.groq_client = None
        
        if not use_local:
            # Kiểm tra API Key
            if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
                logger.warning("GROQ_API_KEY chưa được điền. Tự động chuyển sang dùng Ollama local.")
                self.use_local = True
            else:
                try:
                    self.groq_client = Groq(api_key=GROQ_API_KEY)
                    logger.info(f"Đã kết nối Groq Cloud API. Model mặc định: {GROQ_MODEL}")
                except Exception as e:
                    logger.error(f"Kết nối Groq thất bại: {e}. Chuyển sang dùng Ollama local.")
                    self.use_local = True
                    
        if self.use_local:
            logger.info("Đang khởi chạy LLM qua Ollama local (Yêu cầu ứng dụng Ollama đang chạy).")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sinh câu trả lời từ LLM với cấu hình temperature=0.0 (chống bịa đặt)
        """
        if not self.use_local:
            try:
                # Gọi Groq Cloud API
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=GROQ_MODEL,
                    temperature=0.0, # Ép mô hình sinh kết quả nhất quán nhất
                    max_tokens=1024
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Lỗi gọi Groq API: {e}. Tiến hành fallback sang Ollama...")
                # Nếu Groq lỗi, tự động fallback sang Ollama local
                self.use_local = True
                
        # Gọi Ollama local
        try:
            # Model mặc định là llama3.2 (3B) hoặc phi3, qwen2.5 tùy máy
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
            return "Error: Không thể kết nối tới bất kỳ LLM nào (Groq Cloud lỗi và Ollama Local chưa bật)."
