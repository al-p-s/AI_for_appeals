# make appeal summary by Qwen

import logging
from src.DEMO.loading.qwen_loader import get_qwen

logger = logging.getLogger(__name__)

SUMM_PROMPT = """Кратко изложи суть обращения в 1-2 предложениях: кто обращается, на что жалуется или что просит, и почему.

ОБРАЩЕНИЕ:
{text}"""


def summarize(text: str) -> str:
    logger.info("Qwen summarization started")

    try:
        if len(text) > 3000:
            text = text[:3000]
            logger.info(f"Text shrink up to 6000 symbols")

        prompt = SUMM_PROMPT.format(text=text)

        qwen = get_qwen()

        response = qwen.chat(prompt, temperature=0.2, max_tokens=200)

        if not response:
            logger.warning("Qwen returned empty answer")
            return text[:200] + "..." if len(text) > 200 else text

        summary = response.strip()

        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]
        if summary.startswith("'") and summary.endswith("'"):
            summary = summary[1:-1]

        logger.info(f"Summarization complete")
        return summary

    except Exception as e:
        logger.error(f"Qwen summarization error: {e}")
        return text[:200] + "..." if len(text) > 200 else text
