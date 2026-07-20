# batch pipeline run (run_single.classify_text + NER)
# on test dataset, with metrics calculation, NER and ref. fields

import json
import logging
from collections import defaultdict

from run_single import classify_text

TEST_DATASET_PATH = "../../../data/sets_to_learn/appeals_w_cats/100_for_test_G.json"
TARGET_FIELDS_PATH = "../../../data/sets_to_learn/fields/target_fields_test_100.json"
CATS_L4_PATH = "../../../data/classifier/cats4.json"
ERRORS_OUTPUT_PATH = "../logs/classification_errors.json"

NER_TO_FIELD = {
    "LAST_NAME": "PetitionerSurname",
    "FIRST_NAME": "PetitionerName",
    "MIDDLE_NAME": "PetitionerPatronymic",
    "PERSONAL_EMAIL": "PetitionerEmail",
    "PHONE_NUMBER": "PetitionerPhone",
    "FULL_ADDRESS": "PetitionerAddress",
    "DATE": "PetitionerDate",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("../logs/run_batch_100_REF_by_keryx.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def normalize_codes(categories):
    return [c.split(" ")[0].strip() for c in categories]


def to_l2(code):
    return ".".join(code.split(".")[:2]) + ".0000.0000"


def to_l3(code):
    return ".".join(code.split(".")[:3]) + ".0000"


def parse_pred_codes(pred_text):
    codes = []
    if pred_text.strip():
        for line in pred_text.split("\n"):
            code = line.split(" — ")[0].strip()
            if code:
                codes.append(code)
    return codes


class LevelMetrics:
    def __init__(self):
        self.correct = 0
        self.partial = 0
        self.jaccard_sum = 0.0

    def update(self, pred_set, true_set):
        if pred_set == true_set:
            self.correct += 1
        if pred_set & true_set:
            self.partial += 1
        union = pred_set | true_set
        if union:
            self.jaccard_sum += len(pred_set & true_set) / len(union)

    def report(self, total, name):
        total = total or 1
        logger.info(f"{name} METRICS")
        logger.info(f"  Exact accuracy: {self.correct / total:.4f}")
        logger.info(f"  Partial accuracy: {self.partial / total:.4f}")
        logger.info(f"  Mean Jaccard: {self.jaccard_sum / total:.4f}")


def main():
    logger.info("=" * 60)
    logger.info("Dataset evaluation started")

    with open(TEST_DATASET_PATH, encoding="utf-8") as f:
        appeals = json.load(f)["appeals"]

    with open(TARGET_FIELDS_PATH, encoding="utf-8") as f:
        fields_by_file = {i["file_name"]: i for i in json.load(f)["appeals"]}

    with open(CATS_L4_PATH, encoding="utf-8") as f:
        name_by_code = {c["code"]: c["name"] for c in json.load(f)["categories"]}

    metrics_l2, metrics_l3, metrics_l4 = LevelMetrics(), LevelMetrics(), LevelMetrics()
    total = 0
    errors = []

    ner_correct, ner_total = defaultdict(int), defaultdict(int)
    field_correct, field_total = defaultdict(int), defaultdict(int)

    for idx, item in enumerate(appeals, start=1):
        file_name = item.get("file_name", "unknown")
        text = item.get("text", "")
        true_codes = normalize_codes(item.get("categories", []))
        true_set = set(true_codes)
        true_l2 = {to_l2(c) for c in true_codes}
        true_l3 = {to_l3(c) for c in true_codes}

        logger.info("-" * 60)
        logger.info(f"[{idx}/{len(appeals)}] Processing: {file_name}")
        field_item = fields_by_file.get(file_name, {})

        try:
            summary, _, _, l4_text, field_predictions, entities = classify_text(text)

            pred_codes = parse_pred_codes(l4_text)
            pred_set = set(pred_codes)
            pred_l2 = {to_l2(c) for c in pred_codes}
            pred_l3 = {to_l3(c) for c in pred_codes}

            true_verbose = [f'{c} {name_by_code.get(c, "UNKNOWN")}' for c in true_codes]
            pred_verbose = [f'{c} {name_by_code.get(c, "UNKNOWN")}' for c in pred_codes]
            logger.info(f"TRUE L4: {true_verbose}")
            logger.info(f"PRED L4: {pred_verbose}")

            metrics_l2.update(pred_l2, true_l2)
            metrics_l3.update(pred_l3, true_l3)
            metrics_l4.update(pred_set, true_set)

            logger.info(f"NER: {entities}")

            logger.info("Field predictions:")
            for field_name, pred_value in field_predictions.items():
                gt_value = str(field_item.get(field_name, "")).strip()
                logger.info(f"  {field_name}: TRUE='{gt_value}' | PRED='{pred_value}'")

            for ner_label, field_name in NER_TO_FIELD.items():
                gt_value = str(field_item.get(field_name, "")).strip()
                pred_values = entities.get(ner_label, [])
                if gt_value or pred_values:
                    logger.info(f"  {ner_label}: TRUE='{gt_value}' | PRED={pred_values}")

                if not gt_value:
                    ner_total[ner_label] += 1
                    ner_correct[ner_label] += 1
                    continue

                ner_total[ner_label] += 1
                predicted_lower = [v.lower() for v in pred_values]
                if gt_value.lower() in predicted_lower:
                    ner_correct[ner_label] += 1

            for field_name, pred_value in field_predictions.items():
                gt_value = str(field_item.get(field_name, "")).strip()
                if not gt_value:
                    continue
                field_total[field_name] += 1

                # PetitionerCategory synonyms
                if field_name == "PetitionerCategory":
                    if gt_value.lower() == "другие категории" and str(
                            pred_value).strip().lower() == "категория не установлена":
                        field_correct[field_name] += 1
                        continue
                if str(pred_value).strip().lower() == gt_value.lower():
                    field_correct[field_name] += 1

            if pred_set != true_set:
                errors.append({"file": file_name, "true": list(true_set), "pred": list(pred_set)})

            total += 1

        except Exception as e:
            logger.exception(f"FAILED: {file_name} | Error: {e}")

    logger.info("=" * 60)
    logger.info("FINAL METRICS")
    logger.info(f"Total samples: {total}")
    metrics_l2.report(total, "L2")
    metrics_l3.report(total, "L3")
    metrics_l4.report(total, "L4")
    logger.info(f"  Errors count: {len(errors)}")

    logger.info("=" * 60)
    logger.info("NER FIELD METRICS")
    logger.info(f"{'Class':<25} {'Correct':>10} {'All':>8} {'Accuracy':>10}")
    logger.info("-" * 55)
    total_c, total_t = 0, 0
    for label in sorted(ner_total.keys()):
        c, t = ner_correct[label], ner_total[label]
        total_c += c
        total_t += t
        logger.info(f"{label:<25} {c:>10} {t:>8} {c / t:>10.3f}")
    logger.info("-" * 55)
    logger.info(f"{'FINAL':<25} {total_c:>10} {total_t:>8} {(total_c / total_t if total_t else 0):>10.3f}")

    logger.info("=" * 60)
    logger.info("FIELD CLASSIFICATION METRICS")
    logger.info(f"{'Field':<35} {'Correct':>10} {'All':>8} {'Accuracy':>10}")
    logger.info("-" * 65)
    total_fc, total_ft = 0, 0
    for field_name in sorted(field_total.keys()):
        c, t = field_correct[field_name], field_total[field_name]
        total_fc += c
        total_ft += t
        logger.info(f"{field_name:<35} {c:>10} {t:>8} {c / t:>10.3f}")
    logger.info("-" * 65)
    logger.info(f"{'ИТОГО':<35} {total_fc:>10} {total_ft:>8} {(total_fc / total_ft if total_ft else 0):>10.3f}")

    with open(ERRORS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    logger.info(f"Errors saved to {ERRORS_OUTPUT_PATH}")
    logger.info("Dataset evaluation finished")


if __name__ == "__main__":
    main()
