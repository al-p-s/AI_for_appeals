import logging
import requests
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# LM Studio API configuration
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "qwen/qwen3.5-9b"

class QwenLoader:

    def __init__(self, api_url: str = LM_STUDIO_URL, model: str = MODEL_NAME):
        self.api_url = api_url
        self.model = model
        self._check_connection()
        logger.info(f"Qwen loaded: {model}")

    def _check_connection(self):
        try:
            response = requests.get(
                self.api_url.replace("/chat/completions", "/models"),
                timeout=5
            )
            if response.status_code == 200:
                logger.info("LM Studio API available")
            else:
                logger.warning(f"LM Studio API unavailable: {response.status_code}")
        except Exception as e:
            logger.error(f"LM Studio connection error: {e}")

    def chat(self, prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=90
            )

            if response.status_code != 200:
                logger.error(f"API error: {response.status_code}")
                logger.error(f"Answer: {response.text}")
                return None

            result = response.json()
            content = result["choices"][0]["message"].get("content", "")

            if not content:
                logger.warning("Model returned empty answer")
                return None

            return content

        except requests.exceptions.Timeout:
            logger.error("Model request timeout")
            return None
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

_qwen_instance = None

def get_qwen() -> QwenLoader:
    global _qwen_instance
    if _qwen_instance is None:
        _qwen_instance = QwenLoader()
    return _qwen_instance


logger.info("Qwen is ready")
