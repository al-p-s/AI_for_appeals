# classification funcs
# 1. classify_hierarchy(summary) - cascade multilabel-classification
# by levels L2 -> L3 -> L4

import logging
import torch

from src.DEMO.loading import keryx_loader as kx

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def format_preds(preds):
    return "\n".join(f"{c['code']} {c['name']}" for c, s in preds)


def get_children(cats, parent_code, level):
    prefix = ".".join(parent_code.split(".")[:level])
    return [c for c in cats if c["code"].startswith(prefix + ".")]


def classify_level_multi(summary, candidates, tokenizer_model, threshold, prefix="", batch_size=64):
    tokenizer, model = tokenizer_model
    names = [prefix + c["name"] for c in candidates]

    scores = []
    for i in range(0, len(names), batch_size):
        batch_names = names[i:i + batch_size]
        enc = tokenizer(
            [summary] * len(batch_names), batch_names,
            return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
        scores.extend(probs[:, 0].tolist())

    scored = list(zip(candidates, scores))
    max_score = max(scores)
    result = [(c, s) for c, s in scored if s >= max_score * threshold]

    if not result:
        result = [max(scored, key=lambda x: x[1])]
    return result


def classify_hierarchy(summary: str):
    pred_l2 = classify_level_multi(summary, kx.cats_l2, kx.keryx_l2, kx.THRESHOLD_L2)

    pred_l3 = []
    for l2, l2_score in pred_l2:
        cands_l3 = get_children(kx.cats_l3, l2["code"], 2)
        l2_name = kx.name_by_code.get(l2["code"], "")
        if cands_l3:
            for l3, l3_score in classify_level_multi(
                summary, cands_l3, kx.keryx_l3, kx.THRESHOLD_L3, prefix=f"{l2_name} → "
            ):
                pred_l3.append((l3, l3_score))

    pred_l4 = []
    for l3, l3_score in pred_l3:
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
