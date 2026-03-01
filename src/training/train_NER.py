from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset
from seqeval.metrics import f1_score, classification_report
import json
import numpy as np


dataset = load_dataset("json", data_files={
    "train": "../../data/train_dataset.json",
    "validation": "../../data/validation_dataset.json",
    "test": "../../data/test_dataset.json" } )

model_id = "../../models/NER_RuModernBert_byVK"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision="patched-tokenizer")

with open('../../data/label_mapping.json', 'r', encoding='utf-8') as f:
    label_list = json.load(f)

label2id = label_list["label2id"]
id2label = label_list["id2label"]

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=1024,
        is_split_into_words=True
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned_labels = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                aligned_labels.append(labels[word_id])
            else:
                aligned_labels.append(-100)
            prev_word_id = word_id

        all_labels.append(aligned_labels)

    tokenized["labels"] = all_labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    model_id,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = [
        [label_list[l] for l in label_row if l != -100]
        for label_row in labels
    ]
    true_preds = [
        [label_list[p] for p, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]

    return {"f1": f1_score(true_labels, true_preds)}

training_args = TrainingArguments(
    output_dir="../../models/train_NER_RuModernBert",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()