import json
import logging
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification

logger = logging.getLogger(__name__)

LABEL_MAPPING_PATH = "../../data/label_mapping.json"
MODEL_PATH = "../../models/train_NER_RuModernBert2_0/checkpoint-720"

SKIP_LABELS = {"GOV_EMAIL"}

with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

id2label = {int(k): v for k, v in mapping["id2label"].items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()
model = model.to("cuda")

logger.info("NER model loaded")


def extract_entities(text: str):
    logger.info("NER extraction started")

    words = text.split()

    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    word_ids = inputs.word_ids()
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predictions = logits.argmax(-1)[0].tolist()

    entities = {}
    current_entity = []
    current_label = None

    seen_words = {}

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue

        if word_idx not in seen_words:
            seen_words[word_idx] = id2label[predictions[token_idx]]

    for word_idx, word in enumerate(words):
        label = seen_words.get(word_idx, "O")

        if label.startswith("B-"):
            if current_entity:
                entities.setdefault(current_label, []).append(" ".join(current_entity))

            current_label = label[2:]
            current_entity = [word]

        elif label.startswith("I-") and current_label == label[2:]:
            current_entity.append(word)

        else:
            if current_entity:
                entities.setdefault(current_label, []).append(" ".join(current_entity))

            current_entity = []
            current_label = None

    if current_entity:
        entities.setdefault(current_label, []).append(" ".join(current_entity))

    logger.info(f"NER extracted entities: {entities}")
    logger.info("NER extraction finished")

    return entities


def format_ner_entities(entities: dict):
    if not entities:
        return "Ключевые поля не найдены"

    result = []

    for label, values in entities.items():
        if label in SKIP_LABELS:
            continue
        uniq_values = list(dict.fromkeys(values))
        result.append(f"{label}: {', '.join(uniq_values)}")

    return "\n".join(result)
