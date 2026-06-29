import json
import logging
from collections import defaultdict

from single_inference import classify_text
from ner_inference import extract_entities

TEST_DATASET_PATH = "../../data/sets_to_learn/appeals_w_cats/100_for_test_G.json"
TARGET_FIELDS_PATH = "../../data/sets_to_learn/fields/target_fields_test_100.json"
CATS_L4 = "../../data/classifier/cats4.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/test_100_pipeline_1340v3.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

NER_TO_FIELD = {
    "LAST_NAME": "PetitionerSurname",
    "FIRST_NAME": "PetitionerName",
    "MIDDLE_NAME": "PetitionerPatronymic",
    "PERSONAL_EMAIL": "PetitionerEmail",
    "PHONE_NUMBER": "PetitionerPhone",
    "ADDRESS": "PetitionerAddress",
}


def normalize_codes(categories):
    result = []
    for c in categories:
        code = c.split(" ")[0].strip()
        result.append(code)
    return result


def main():
    logger.info("=" * 60)
    logger.info("Dataset evaluation started")

    with open(TEST_DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    appeals = data["appeals"]

    with open(TARGET_FIELDS_PATH, encoding="utf-8") as f:
        target_data = json.load(f)
    fields_by_file = {i["file_name"]: i for i in target_data["appeals"]}

    with open(CATS_L4, encoding="utf-8") as f:
        cats_l4 = json.load(f)["categories"]
    name_by_code = {
        c["code"]: c["name"]
        for c in cats_l4
    }

    total = 0
    correct_l4 = 0
    partial_l4 = 0
    jaccard_sum = 0.0

    correct_l3 = 0
    partial_l3 = 0
    jaccard_l3_sum = 0.0

    correct_l2 = 0
    partial_l2 = 0
    jaccard_l2_sum = 0.0

    errors = []

    ner_correct = defaultdict(int)
    ner_total = defaultdict(int)

    field_correct = defaultdict(int)
    field_total = defaultdict(int)

    for idx, item in enumerate(appeals, start=1):

        file_name = item.get("file_name", "unknown")
        text = item.get("text", "")
        true_categories = normalize_codes(item.get("categories", []))
        true_l2 = {
            ".".join(c.split(".")[:2]) + ".0000.0000"
            for c in true_categories
        }

        true_l3 = {
            ".".join(c.split(".")[:3]) + ".0000"
            for c in true_categories
        }

        logger.info("-" * 60)
        logger.info(f"[{idx}/{len(appeals)}] Processing: {file_name}")
        field_item = fields_by_file.get(file_name, {})

        try:
            summary, pred_l2, pred_l3, pred_l4, field_predictions = classify_text(text)

            pred_codes = []
            if pred_l4.strip():
                for line in pred_l4.split("\n"):
                    code = line.split(" — ")[0].strip()
                    pred_codes.append(code)

            pred_l2_set = {
                ".".join(c.split(".")[:2]) + ".0000.0000"
                for c in pred_codes
            }

            pred_l3_set = {
                ".".join(c.split(".")[:3]) + ".0000"
                for c in pred_codes
            }

            pred_set = set(pred_codes)
            true_set = set(true_categories)
            true_verbose = [
                f"{c} {name_by_code.get(c, 'UNKNOWN')}"
                for c in true_categories
            ]
            pred_verbose = [
                f"{c} {name_by_code.get(c, 'UNKNOWN')}"
                for c in pred_codes
            ]
            logger.info(f"TRUE L4: {true_verbose}")
            logger.info(f"PRED L4: {pred_verbose}")

            # L2 exact
            if pred_l2_set == true_l2:
                correct_l2 += 1

            # L2 partial
            if pred_l2_set & true_l2:
                partial_l2 += 1

            # L2 jaccard
            union_l2 = pred_l2_set | true_l2
            inter_l2 = pred_l2_set & true_l2
            if union_l2:
                jaccard_l2_sum += len(inter_l2) / len(union_l2)

            # L3 exact
            if pred_l3_set == true_l3:
                correct_l3 += 1

            # L3 partial
            if pred_l3_set & true_l3:
                partial_l3 += 1

            # L3 jaccard
            union_l3 = pred_l3_set | true_l3
            inter_l3 = pred_l3_set & true_l3
            if union_l3:
                jaccard_l3_sum += len(inter_l3) / len(union_l3)

            # L4 exact
            if pred_set == true_set:
                correct_l4 += 1

            # L4 partial
            if pred_set & true_set:
                partial_l4 += 1

            # L4 jaccard
            union = pred_set | true_set
            inter = pred_set & true_set
            if union:
                jaccard_sum += len(inter) / len(union)

            # NER
            entities = extract_entities(text)
            logger.info(f"NER: {entities}")
            # Field predictions vs true
            logger.info("Field predictions:")
            for field_name, pred_value in field_predictions.items():
                gt_value = field_item.get(field_name, "").strip()
                logger.info(f"  {field_name}: TRUE='{gt_value}' | PRED='{pred_value}'")
            # NER true vs pred
            for ner_label, field_name in NER_TO_FIELD.items():
                gt_value = field_item.get(field_name, "").strip()
                pred_values = entities.get(ner_label, [])
                if gt_value or pred_values:
                    logger.info(f"  {ner_label}: TRUE='{gt_value}' | PRED={pred_values}")

            # NER field metrics
            for ner_label, field_name in NER_TO_FIELD.items():
                gt_value = field_item.get(field_name, "").strip().lower()
                if not gt_value:
                    continue
                ner_total[ner_label] += 1
                predicted_values = [v.lower() for v in entities.get(ner_label, [])]
                if any(gt_value == pred for pred in predicted_values):
                    ner_correct[ner_label] += 1

            for field_name, pred_value in field_predictions.items():
                gt_value = field_item.get(field_name, "").strip()
                if not gt_value:
                    continue
                field_total[field_name] += 1
                if pred_value.strip().lower() == gt_value.lower():
                    field_correct[field_name] += 1

            # errors
            if pred_set != true_set:
                errors.append({
                    "file": file_name,
                    "true": list(true_set),
                    "pred": list(pred_set)
                })

            total += 1

        except Exception as e:
            logger.exception(f"FAILED: {file_name}")

    # === CLASSIFICATION METRICS ===
    exact_acc = correct_l4 / total if total else 0
    partial_acc = partial_l4 / total if total else 0
    jaccard_avg = jaccard_sum / total if total else 0

    exact_l2 = correct_l2 / total if total else 0
    partial_acc_l2 = partial_l2 / total if total else 0
    jaccard_avg_l2 = jaccard_l2_sum / total if total else 0

    exact_l3 = correct_l3 / total if total else 0
    partial_acc_l3 = partial_l3 / total if total else 0
    jaccard_avg_l3 = jaccard_l3_sum / total if total else 0

    logger.info("=" * 60)
    logger.info("FINAL METRICS")
    logger.info(f"Total samples: {total}")

    logger.info("L2 METRICS")
    logger.info(f"  Exact L2 accuracy: {exact_l2:.4f}")
    logger.info(f"  Partial L2 accuracy: {partial_acc_l2:.4f}")
    logger.info(f"  Mean L2 Jaccard: {jaccard_avg_l2:.4f}")

    logger.info("L3 METRICS")
    logger.info(f"  Exact L3 accuracy: {exact_l3:.4f}")
    logger.info(f"  Partial L3 accuracy: {partial_acc_l3:.4f}")
    logger.info(f"  Mean L3 Jaccard: {jaccard_avg_l3:.4f}")

    logger.info("L4 METRICS")
    logger.info(f"  Exact L4 accuracy: {exact_acc:.4f}")
    logger.info(f"  Partial L4 accuracy: {partial_acc:.4f}")
    logger.info(f"  Mean L4 Jaccard: {jaccard_avg:.4f}")
    logger.info(f"  Errors count: {len(errors)}")

    # === NER METRICS ===
    logger.info("=" * 60)
    logger.info("NER FIELD METRICS")
    logger.info(f"{'Класс':<25} {'Правильно':>10} {'Всего':>8} {'Accuracy':>10}")
    logger.info("-" * 55)
    total_c, total_t = 0, 0
    for label in sorted(ner_total.keys()):
        c = ner_correct[label]
        t = ner_total[label]
        total_c += c
        total_t += t
        logger.info(f"{label:<25} {c:>10} {t:>8} {c/t:>10.3f}")
    logger.info("-" * 55)
    overall = total_c / total_t if total_t else 0
    logger.info(f"{'ИТОГО':<25} {total_c:>10} {total_t:>8} {overall:>10.3f}")

    logger.info("=" * 60)
    logger.info("FIELD CLASSIFICATION METRICS")
    logger.info(f"{'Поле':<35} {'Правильно':>10} {'Всего':>8} {'Accuracy':>10}")
    logger.info("-" * 65)
    total_fc, total_ft = 0, 0
    for field_name in sorted(field_total.keys()):
        c = field_correct[field_name]
        t = field_total[field_name]
        total_fc += c
        total_ft += t
        logger.info(f"{field_name:<35} {c:>10} {t:>8} {c / t:>10.3f}")
    logger.info("-" * 65)
    overall_f = total_fc / total_ft if total_ft else 0
    logger.info(f"{'ИТОГО':<35} {total_fc:>10} {total_ft:>8} {overall_f:>10.3f}")

    with open("logs/classification_errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    logger.info("Errors saved to logs/classification_errors.json")
    logger.info("Dataset evaluation finished")


if __name__ == "__main__":
    main()
