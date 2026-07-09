# classification funcs
# 1. classify_hierarchy(summary) - cascade multilabel-classification
# by levels L2 -> L3 -> L4
# 2. classify_all_fields(text) - additional fileds classification
# (AppealKind, ItemID, DeliveryTypeId, RegistrationPlaceId, PetitionerDistrict,
# PetitionerCategory, StatusId, ConsiderationType), each field - separate model.

import logging

import torch

from src.DEMO.loading import keryx_loader as kx

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def format_preds(preds):
    return "\n".join(f"{c['code']} — {c['name']}" for c, s in preds)


def get_children(cats, parent_code, level):
    prefix = ".".join(parent_code.split(".")[:level])
    return [c for c in cats if c["code"].startswith(prefix + ".")]


def classify_level_multi(summary, candidates, tokenizer_model, threshold, prefix=""):
    tokenizer, model = tokenizer_model
    scores = []
    logger.info(f"Classification started | candidates={len(candidates)} | threshold={threshold}")

    for c in candidates:
        enc = tokenizer(summary, prefix + c["name"], return_tensors="pt",
                         truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
        scores.append((c, probs[0][0].item()))

    max_score = max(s for _, s in scores)
    result = [(c, s) for c, s in scores if s >= max_score * threshold]
    logger.info(f"Classification finished | selected={len(result)}")

    if not result:
        result = [max(scores, key=lambda x: x[1])]
    return result


def classify_hierarchy(summary: str):
    pred_l2 = classify_level_multi(summary, kx.cats_l2, kx.keryx_l2, kx.THRESHOLD_L2)

    pred_l3 = []
    for l2, l2_score in pred_l2:
        if l2_score < kx.ABS_THRESHOLD_L2:
            logger.info(f"Skip L3 for {l2['code']} (score={l2_score:.3f})")
            continue
        cands_l3 = get_children(kx.cats_l3, l2["code"], 2)
        l2_name = kx.name_by_code.get(l2["code"], "")
        if cands_l3:
            for l3, l3_score in classify_level_multi(
                summary, cands_l3, kx.keryx_l3, kx.THRESHOLD_L3, prefix=f"{l2_name} → "
            ):
                pred_l3.append((l3, l3_score))

    pred_l4 = []
    for l3, l3_score in pred_l3:
        if l3_score < kx.ABS_THRESHOLD_L3:
            logger.info(f"Skip L4 for {l3['code']} (score={l3_score:.3f})")
            continue
        cands_l4 = get_children(kx.cats_l4, l3["code"], 3)
        l3_name = kx.name_by_code.get(l3["code"], "")
        l2_code = ".".join(l3["code"].split(".")[:2]) + ".0000.0000"
        l2_name = kx.name_by_code.get(l2_code, "")
        if cands_l4:
            for l4, l4_score in classify_level_multi(
                summary, cands_l4, kx.keryx_l4, kx.THRESHOLD_L4,
                prefix=f"{l2_name} → {l3_name} → "
            ):
                pred_l4.append((l4, l4_score))

    return pred_l2, pred_l3, pred_l4


def classify_field(text, candidates, keryx_model):
    tokenizer, model = keryx_model
    scores = []
    for candidate in candidates:
        enc = tokenizer(text, candidate, return_tensors="pt",
                         truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits
        scores.append(logits[0, 0].item())
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return candidates[best_idx]


def classify_all_fields(text: str) -> dict:
    predictions = {}
    for field_name, (keryx_model, candidates) in kx.field_models.items():
        predictions[field_name] = classify_field(text, candidates, keryx_model)
    logger.info(f"Field predictions: {predictions}")
    return predictions
