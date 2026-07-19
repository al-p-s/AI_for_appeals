# src/DEMO/loading/keryx_loader.py

# loading all KERYX-models (hierarchical classification L2/L3/L4)

import json
import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# paths
CATS_L2_PATH = "../../../data/classifier/cats2.json"
CATS_L3_PATH = "../../../data/classifier/cats3.json"
CATS_L4_PATH = "../../../data/classifier/cats4.json"

KERYX_PATH_L2 = "../../../models/KERYX_1340_G/L2"
KERYX_PATH_L3 = "../../../models/KERYX_1340_G/L3"
KERYX_PATH_L4 = "../../../models/KERYX_1340_G/L4"

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
    model = AutoModelForSequenceClassification.from_pretrained(path).to("cuda").eval()
    return tokenizer, model


logger.info("KERYX loader: loading category dictionaries and models...")

cats_l2 = load_json(CATS_L2_PATH)["categories"]
cats_l3 = load_json(CATS_L3_PATH)["categories"]
cats_l4 = load_json(CATS_L4_PATH)["categories"]
name_by_code = {c["code"]: c["name"] for c in cats_l2 + cats_l3 + cats_l4}

keryx_l2 = load_keryx(KERYX_PATH_L2)
keryx_l3 = load_keryx(KERYX_PATH_L3)
keryx_l4 = load_keryx(KERYX_PATH_L4)

logger.info("KERYX loader: all models ready.")
