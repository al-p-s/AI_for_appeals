import logging
import torch

from src.DEMO.loading import keryx_loader as kx

logger = logging.getLogger(__name__)

CONSIDERATION_TYPE_MAPPING = {
    "первичное": "0",
    "повторное": "1",
    "неоднократное": "2",
}


def classify_field(text, candidates, keryx_model, batch_size=64):
    tokenizer, model = keryx_model
    scores = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        enc = tokenizer(
            [text] * len(batch), batch,
            return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits
        scores.extend(logits[:, 0].tolist())
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return candidates[best_idx]


def classify_all_fields(text: str) -> dict:
    predictions = {}
    for field_name, (keryx_model, candidates) in kx.field_models.items():
        predictions[field_name] = classify_field(text, candidates, keryx_model)

    if "ConsiderationType" in predictions:
        value = predictions["ConsiderationType"]
        value_lower = value.lower().strip()
        if value_lower in CONSIDERATION_TYPE_MAPPING:
            predictions["ConsiderationType"] = CONSIDERATION_TYPE_MAPPING[value_lower]

    logger.info(f"Field predictions: {predictions}")
    return predictions
