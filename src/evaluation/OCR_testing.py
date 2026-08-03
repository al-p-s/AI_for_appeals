import json
import re
import argparse
from jiwer import cer, wer


def normalize_for_comparison(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_appeals(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['file_name']: item['text'] for item in data['appeals']}


def compare(ground_truth_path, ocr_result_path, output_path=None, cer_threshold=None):
    ground_truth = load_appeals(ground_truth_path)
    ocr_result = load_appeals(ocr_result_path)

    results = []
    missing_in_ocr = []

    for file_name, gt_text in ground_truth.items():
        if file_name not in ocr_result:
            missing_in_ocr.append(file_name)
            continue

        ocr_text = ocr_result[file_name]

        gt_norm = normalize_for_comparison(gt_text)
        ocr_norm = normalize_for_comparison(ocr_text)

        if not gt_norm:
            continue  # пустой эталон - нечего сравнивать

        score_cer = cer(gt_norm, ocr_norm)
        score_wer = wer(gt_norm, ocr_norm)

        results.append({
            "file_name": file_name,
            "cer": round(score_cer, 4),
            "wer": round(score_wer, 4),
            "gt_length": len(gt_norm),
            "ocr_length": len(ocr_norm),
        })

    results.sort(key=lambda x: x["cer"], reverse=True)

    if missing_in_ocr:
        print(f"ВНИМАНИЕ: {len(missing_in_ocr)} документов есть в эталоне, но отсутствуют в результате OCR:")
        for fn in missing_in_ocr:
            print(f"  - {fn}")
        print()

    if not results:
        print("Нет документов для сравнения.")
        return

    avg_cer = sum(r["cer"] for r in results) / len(results)
    avg_wer = sum(r["wer"] for r in results) / len(results)

    sorted_cer = sorted(r["cer"] for r in results)
    median_cer = sorted_cer[len(sorted_cer) // 2]

    print(f"Сравнено документов: {len(results)}")
    print(f"Средний CER: {avg_cer:.4f}   Медианный CER: {median_cer:.4f}")
    print(f"Средний WER: {avg_wer:.4f}")
    print()
    print("Худшие 5 документов по CER:")
    for r in results[:5]:
        print(f"  {r['file_name']}: CER={r['cer']:.4f}  WER={r['wer']:.4f}")

    if cer_threshold is not None:
        failed = [r for r in results if r["cer"] > cer_threshold]
        print(f"\nДокументов с CER выше порога {cer_threshold}: {len(failed)} из {len(results)}")
        for r in failed:
            print(f"  {r['file_name']}: CER={r['cer']:.4f}")

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "summary": {
                        "documents_compared": len(results),
                        "avg_cer": round(avg_cer, 4),
                        "median_cer": round(median_cer, 4),
                        "avg_wer": round(avg_wer, 4),
                        "missing_in_ocr": missing_in_ocr,
                    },
                    "per_document": results,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )
        print(f"\nПодробный отчёт сохранён в {output_path}")


if __name__ == "__main__":
    compare("../../data/OCR_improving/OCR_test.json", "../../data/OCR_improving/appeals121_20_06_TEST4.json", "report_GLM300.json", 0.1)
