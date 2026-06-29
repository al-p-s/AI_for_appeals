import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json

with open("../../data/label_mapping.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)
id2label = {int(k): v for k, v in mapping["id2label"].items()}

model_path = "../../models/train_NER_RuModernBert2_0/checkpoint-720"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

def predict_ner(text: str):
    words = text.split()
    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    word_ids = inputs.word_ids()

    with torch.no_grad():
        logits = model(**inputs).logits

    predictions = logits.argmax(-1)[0].tolist()

    seen_words = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx not in seen_words:
            seen_words[word_idx] = id2label[predictions[token_idx]]

    print(f"{'Слово':<20} {'Метка'}")
    print("-" * 35)
    for word_idx, word in enumerate(words):
        label = seen_words.get(word_idx, "O")
        print(f"{word:<20} {label}")

predict_ner("Иванов Пётр Сергеевич отправил письмо на адрес ivanov@mail.ru, его телефон 79201111111, живет он по адресу г.Челябинск, ул.Ленина, д.10")