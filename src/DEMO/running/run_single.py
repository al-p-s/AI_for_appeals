# single run of full appeal-processing pipeline
# OCR (GLM-OCR) -> summarization -> hierarchical classification L2/L3/L4 -> reference fields classification

import re
import logging
from pathlib import Path

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     handlers=[
#         logging.FileHandler("../logs/run_single.log", encoding="utf-8"),
#         logging.StreamHandler()
#     ]
# )
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from src.DEMO.functional.qwen_REF_classification import classify_fields_by_qwen
from src.DEMO.text_extraction.text_extraction_glm_ocr import extract_text_from_pdf
from src.DEMO.functional.qwen_make_summary import summarize
from src.DEMO.functional.keryx_classifier import classify_hierarchy, format_preds
from src.DEMO.functional.keryx_REF_classification import classify_all_fields
from src.DEMO.functional.qwen_NER_inference import extract_entities

PDF_FILE_NAME = "../268-5.pdf"
PDF_PATH = Path(__file__).resolve().parent / PDF_FILE_NAME


def _clean_email(value: str) -> str:
    value = re.sub(r"\s+", "", value)
    value = value.strip(" ()[]{}:;,.")
    return value


def _clean_name(value: str) -> str:
    value = value.strip(" .")
    if not value:
        return value
    if " " in value:
        value = value.split()[0]
    value = re.split(r"(?<=[а-яёa-z])(?=[А-ЯЁA-Z])", value)[0]
    return value.capitalize()


def _clean_addr_part(value: str) -> str:
    return value.strip(" .,")


NER_CLEANERS = {
    "GOV_EMAIL": _clean_email,
    "PERSONAL_EMAIL": _clean_email,
    "LAST_NAME": _clean_name,
    "FIRST_NAME": _clean_name,
    "MIDDLE_NAME": _clean_name,
    "POSTAL_CODE": _clean_addr_part,
    "REGION": _clean_addr_part,
    "CITY": _clean_addr_part,
    "STREET": _clean_addr_part,
    "HOUSE": _clean_addr_part,
    "ROOM": _clean_addr_part,
}


def postprocess_entities(entities: dict) -> dict:
    for label, cleaner in NER_CLEANERS.items():
        if label in entities and entities[label]:
            if isinstance(entities[label], list):
                cleaned = [cleaner(v) for v in entities[label] if v]
                entities[label] = list(dict.fromkeys(cleaned))
            else:
                entities[label] = cleaner(entities[label])

    address_parts = []
    address_order = ["POSTAL_CODE", "REGION", "CITY", "STREET", "HOUSE", "ROOM"]

    for key in address_order:
        if key in entities and entities[key]:
            value = entities[key]
            if isinstance(value, list) and value:
                address_parts.append(value[0])
            elif isinstance(value, str):
                address_parts.append(value)

    if address_parts:
        entities["FULL_ADDRESS"] = [", ".join(address_parts)]

    logger.info(f"Postprocessed entities: {entities}")
    return entities


def classify_text(text: str):
    summary = summarize(text)

    pred_l2, pred_l3, pred_l4 = classify_hierarchy(summary)

    logger.info(f"Summary: {summary}")
    logger.info(f"L2 predictions: {[(c['code'], c['name'], round(s, 3)) for c, s in pred_l2]}")
    logger.info(f"L3 predictions: {[(c['code'], c['name'], round(s, 3)) for c, s in pred_l3]}")
    logger.info(f"L4 predictions: {[(c['code'], c['name'], round(s, 3)) for c, s in pred_l4]}")

    field_predictions = classify_all_fields(text)
    # field_predictions = classify_fields_by_qwen(text)

    entities = extract_entities(text)
    entities = postprocess_entities(entities)

    return (
        summary,
        format_preds(pred_l2),
        format_preds(pred_l3),
        format_preds(pred_l4),
        field_predictions,
        entities,
    )


def classify_text_from_pdf(pdf_path, dpi: int = None):
    pdf_path = str(pdf_path)
    logger.info(f"OCR started: {pdf_path}")
    if dpi is not None:
        text = extract_text_from_pdf(pdf_path, dpi=dpi)
    else:
        text = extract_text_from_pdf(pdf_path)
    logger.info(f"OCR finished: {pdf_path} | {len(text)} symbols")

    with open("ocr_150dpi_debug.txt", "w", encoding="utf-8") as f:
        f.write(text)

    return classify_text(text)


def main():
    logger.info("=" * 55)
    logger.info(f"Inference on: {PDF_PATH}")

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    summary, l2, l3, l4, fields, entities = classify_text_from_pdf(PDF_PATH)
    logger.info("Done")


if __name__ == "__main__":
    main()
