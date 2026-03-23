import json
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer
from geracl import GeraclHF, ZeroShotClassificationPipeline
from src.embeddings.USER2Embeddings import USER2Embeddings

CHROMA_PATH = "../../data/chroma"
COLLECTION_NAME = "appeals_cats"
GERACL_MODEL = "../../models/GeRaCl-USER2-base"
APPEALS_WITH_CATS = "../../data/appeals_with_cats.json"
APPEALS_WITH_TEXT = "../../data/appeals.json"
TOP_K = 10

MARKERS = [
    "Текст обращения:",
    "Текст обращения :",
    "Содержание обращения:",
    "Содержание обращения :",
    "Вопрос:"
]


def load_chroma() -> Chroma:
    embeddings = USER2Embeddings()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def load_geracl() -> ZeroShotClassificationPipeline:
    model = GeraclHF.from_pretrained(GERACL_MODEL).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(GERACL_MODEL)
    return ZeroShotClassificationPipeline(model, tokenizer, device="cuda")


def extract_content(text: str) -> tuple[str, str]:
    for marker in MARKERS:
        idx = text.find(marker)
        if idx != -1:
            extracted = text[idx + len(marker):].strip()
            return extracted, marker
    return text, ""


def get_true_codes(categories: list[str]) -> tuple[set[str], list[dict]]:
    codes = set()
    labeled = []
    for cat in categories:
        parts = cat.split(" ", 1)
        code = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else ""
        codes.add(code)
        labeled.append({"code": code, "name": name})
    return codes, labeled


def classify(text: str, chroma_db: Chroma, geracl_pipe: ZeroShotClassificationPipeline) -> tuple[str, list[str], list[dict], int]:
    results = chroma_db.similarity_search_with_score(text, k=TOP_K)

    candidates = []
    for doc, score in results:
        candidates.append({
            "code": doc.metadata["code"],
            "name": doc.metadata["name"],
            "label": doc.metadata["name"],
        })

    labels = [c["label"] for c in candidates]
    best_idx = geracl_pipe([text], [labels], batch_size=1)[0]
    predicted_code = candidates[best_idx]["code"]
    top_k_codes = [c["code"] for c in candidates]

    return predicted_code, top_k_codes, candidates, best_idx


def evaluate():
    with open(APPEALS_WITH_CATS, "r", encoding="utf-8") as f:
        cats_data = json.load(f)
    with open(APPEALS_WITH_TEXT, "r", encoding="utf-8") as f:
        text_data = json.load(f)

    text_by_file = {a["file_name"]: a["text"] for a in text_data["appeals"]}
    appeals = cats_data["appeals"]

    print(f"Всего обращений с категориями: {len(appeals)}")
    print(f"Обращений с текстом: {len(text_by_file)}")
    print("Загружаем модели...")
    chroma_db = load_chroma()
    geracl_pipe = load_geracl()
    print("Готово.\n")

    # статистика маркеров
    marker_found = 0
    marker_stats = {m: 0 for m in MARKERS}

    # метрики полный текст
    full_strict = 0
    full_topk = 0

    # метрики обрезанный текст
    cut_strict = 0
    cut_topk = 0

    total = 0
    skipped = 0
    errors_cut = []

    for appeal in tqdm(appeals, desc="Evaluating"):
        file_name = appeal["file_name"]
        text = text_by_file.get(file_name, "").strip()

        if not text:
            skipped += 1
            continue

        true_codes, true_labeled = get_true_codes(appeal["categories"])
        extracted, found_marker = extract_content(text)

        if found_marker:
            marker_found += 1
            marker_stats[found_marker] += 1

        try:
            pred_full, topk_full, _, _ = classify(text, chroma_db, geracl_pipe)
            pred_cut, topk_cut, candidates_cut, best_idx_cut = classify(extracted, chroma_db, geracl_pipe)
        except Exception as e:
            print(f"Ошибка на {file_name}: {e}")
            skipped += 1
            continue

        total += 1

        if pred_full in true_codes:
            full_strict += 1
        if true_codes & set(topk_full):
            full_topk += 1

        if pred_cut in true_codes:
            cut_strict += 1
        if true_codes & set(topk_cut):
            cut_topk += 1
        else:
            errors_cut.append({
                "file": file_name,
                "true": true_labeled,
                "predicted_code": pred_cut,
                "predicted_label": candidates_cut[best_idx_cut]["label"],
                "top_k": [{"code": c["code"], "label": c["label"]} for c in candidates_cut],
            })

    print(f"\n{'='*55}")
    print(f"Обработано: {total}  |  Пропущено: {skipped}")
    print(f"Маркер найден: {marker_found}/{total} ({marker_found/total*100:.1f}%)")
    print(f"\nСтатистика по маркерам:")
    for m, cnt in marker_stats.items():
        if cnt > 0:
            print(f"  '{m}': {cnt}")
    print(f"\n{'='*55}")
    print(f"{'':30} {'Полный':>10} {'Обрезанный':>12}")
    print(f"{'Strict accuracy':30} {full_strict/total*100:>9.1f}% {cut_strict/total*100:>11.1f}%")
    print(f"{'Top-{} accuracy'.format(TOP_K):30} {full_topk/total*100:>9.1f}% {cut_topk/total*100:>11.1f}%")
    print(f"{'='*55}")

    if errors_cut:
        with open("../../data/eval_errors_cut.json", "w", encoding="utf-8") as f:
            json.dump(errors_cut, f, ensure_ascii=False, indent=2)
        print(f"\nКейсы где топ-{TOP_K} не попал (обрезанный): {len(errors_cut)} шт. → data/eval_errors_cut.json")


if __name__ == "__main__":
    evaluate()
