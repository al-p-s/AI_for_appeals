# src/DEMO/loading/keryx_loader.py

# loading all KERYX-models (hierarchical classification L2/L3/L4)

import json
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.DEMO.paths_config import (
    CATS_L2_PATH,
    CATS_L3_PATH,
    CATS_L4_PATH,
    KERYX_PATH_L2,
    KERYX_PATH_L3,
    KERYX_PATH_L4,
    FIELDS_CONFIG_PATH,
    KERYX_FIELDS_DIR_TEMPLATE as KERYX_FIELDS_DIR,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# thresholds
THRESHOLD_L2 = 0.9
THRESHOLD_L3 = 0.8
THRESHOLD_L4 = 0.8

# # high-level cats scores
# ABS_THRESHOLD_L2 = 0.1  # lower -> don't go to L3
# ABS_THRESHOLD_L3 = 0.1  # lower -> don't go to L4


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_keryx(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, torch_dtype=torch.float16
    ).to("cuda").eval()
    return tokenizer, model

def load_fields_config(path=FIELDS_CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_field_models(fields_config):
    models = {}
    for cfg in fields_config:
        field_name = cfg["field_name"]
        path = KERYX_FIELDS_DIR.format(field_name)
        try:
            models[field_name] = (load_keryx(path), cfg["field_candidates"])
            logger.info(f"Field model loaded: {field_name}")
        except Exception as e:
            logger.warning(f"Field model not found: {field_name} | {e}")
    return models


_loaded = False
_cats_l2 = None
_cats_l3 = None
_cats_l4 = None
_name_by_code = None
_keryx_l2 = None
_keryx_l3 = None
_keryx_l4 = None
_field_models = None


def load_keryx_models():
    global _loaded, _cats_l2, _cats_l3, _cats_l4, _name_by_code
    global _keryx_l2, _keryx_l3, _keryx_l4, _field_models

    if _loaded:
        return

    logger.info("Loading KERYX taxonomy dictionaries...")
    _cats_l2 = load_json(CATS_L2_PATH)["categories"]
    _cats_l3 = load_json(CATS_L3_PATH)["categories"]
    _cats_l4 = load_json(CATS_L4_PATH)["categories"]
    _name_by_code = {c["code"]: c["name"] for c in _cats_l2 + _cats_l3 + _cats_l4}

    logger.info("Loading KERYX hierarchy models (L2, L3, L4) to CUDA...")
    _keryx_l2 = load_keryx(KERYX_PATH_L2)
    _keryx_l3 = load_keryx(KERYX_PATH_L3)
    _keryx_l4 = load_keryx(KERYX_PATH_L4)

    logger.info("Loading KERYX reference field models to CUDA...")
    fields_config = load_fields_config()
    _field_models = load_field_models(fields_config)

    _loaded = True
    logger.info("KERYX loader: all models ready.")


def __getattr__(name: str):
    """PEP 562: deferred access to module-level model attributes."""
    attrs = {
        "cats_l2",
        "cats_l3",
        "cats_l4",
        "name_by_code",
        "keryx_l2",
        "keryx_l3",
        "keryx_l4",
        "field_models",
    }
    if name in attrs:
        load_keryx_models()
        return globals()[f"_{name}"]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

