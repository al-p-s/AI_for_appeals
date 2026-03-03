import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
from seqeval.scheme import IOB2
from seqeval.metrics.sequence_labeling import get_entities
from collections import defaultdict

with open("../../data/label_mapping.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)

id2label = {int(k): v for k, v in mapping["id2label"].items()}
label2id = mapping["label2id"]

model_path = "../../models/train_NER_RuModernBert/checkpoint-400"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()
model = model.to("cuda")

dataset = load_dataset("json", data_files={"test": "../../data/test_dataset.json"})


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels


def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=1024,
        is_split_into_words=True,
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        all_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized["labels"] = all_labels
    return tokenized


tokenized_test = dataset["test"].map(tokenize_and_align_labels, batched=True)

keep_columns = ["input_ids", "attention_mask", "labels"]
remove_columns = [col for col in tokenized_test.column_names if col not in keep_columns]
tokenized_test = tokenized_test.remove_columns(remove_columns)
tokenized_test.set_format("torch")

data_collator = DataCollatorForTokenClassification(tokenizer)
dataloader = DataLoader(tokenized_test, batch_size=16, collate_fn=data_collator)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dataloader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()

        for pred_row, label_row in zip(predictions, labels):
            preds = [id2label[int(p)] for p, l in zip(pred_row, label_row) if l != -100]
            labs  = [id2label[int(l)] for l in label_row if l != -100]
            all_preds.append(preds)
            all_labels.append(labs)

print("\nотчёт:")
print(classification_report(all_labels, all_preds, mode="strict", scheme=IOB2))

correct_per_class = defaultdict(int)
total_per_class = defaultdict(int)

for label_seq, pred_seq in zip(all_labels, all_preds):
    # get_entities возвращает список (тип, начало, конец)
    # например: [("LAST_NAME", 0, 1), ("FIRST_NAME", 2, 3)]
    true_entities = get_entities(label_seq, suffix=False)
    pred_entities = get_entities(pred_seq, suffix=False)

    pred_set = set(pred_entities)  # для быстрого поиска

    for entity in true_entities:
        entity_type = entity[0]   # "LAST_NAME"
        total_per_class[entity_type] += 1

        if entity in pred_set:    # совпадает тип И границы
            correct_per_class[entity_type] += 1

print("\nAccuracy на уровне сущностей:")
print(f"{'Класс':<25} {'Правильно':>10} {'Всего':>8} {'Accuracy':>10}")
print("-" * 55)

total_correct = 0
total_all = 0

for label in sorted(total_per_class.keys()):
    total = total_per_class[label]
    correct = correct_per_class[label]
    acc = correct / total if total > 0 else 0
    total_correct += correct
    total_all += total
    print(f"{label:<25} {correct:>10} {total:>8} {acc:>10.3f}")

print("-" * 55)
overall = total_correct / total_all if total_all > 0 else 0
print(f"{'ИТОГО':<25} {total_correct:>10} {total_all:>8} {overall:>10.3f}")