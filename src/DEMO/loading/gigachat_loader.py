# loading GigaChat-Lite and appeal-text summarization

import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from src.DEMO.paths_config import GIGACHAT_PATH

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SUMM_PROMPT = """Кратко изложи суть обращения в 1-2 предложениях: кто обращается, на что жалуется или что просит, и почему.
ОБРАЩЕНИЕ:
{text}"""

def load_gigachat():
    tokenizer = AutoTokenizer.from_pretrained(GIGACHAT_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        GIGACHAT_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    generation_config = GenerationConfig.from_pretrained(GIGACHAT_PATH, trust_remote_code=True)
    generation_config.do_sample = False
    return tokenizer, model, generation_config


logger.info("GigaChat loader: loading model...")
gigachat_tok, gigachat_model, gigachat_gen = load_gigachat()
logger.info("GigaChat loader: model ready.")


def summarize(text: str) -> str:
    prompt = gigachat_tok.apply_chat_template(
        [{"role": "user", "content": SUMM_PROMPT.format(text=text[:3000])}],
        tokenize=False, add_generation_prompt=True
    )
    data = gigachat_tok(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(gigachat_model.device) for k, v in data.items()}
    data.pop("token_type_ids", None)

    output_ids = gigachat_model.generate(**data, generation_config=gigachat_gen)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    return gigachat_tok.decode(output_ids, skip_special_tokens=True).strip()
