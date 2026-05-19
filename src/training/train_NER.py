from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
import json
import numpy as np

dataset = load_dataset("json", data_files={
    "train": "../../data/train_dataset.json",
    "validation": "../../data/validation_dataset.json",
    "test": "../../data/test_dataset.json"
})

model_id = "../../models/NER_RuModernBert_byVK"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision="patched-tokenizer")

with open('../../data/label_mapping.json', 'r', encoding='utf-8') as f:
    label_list = json.load(f)

label2id = label_list["label2id"]
id2label = {int(k): v for k, v in label_list["id2label"].items()}


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
        is_split_into_words=True
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        all_labels.append(align_labels_with_tokens(labels, word_ids))
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
        [id2label[int(l)] for l in label_row if l != -100]
        for label_row in labels
    ]
    true_preds = [
        [id2label[int(p)] for p, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]
    return {"f1": f1_score(true_labels, true_preds, mode="strict", scheme=IOB2)}

training_args = TrainingArguments(
    output_dir="../../models/train_NER_RuModernBert",
    num_train_epochs=6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    weight_decay=0.005,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
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

results = trainer.evaluate(tokenized_dataset["test"])
print(results)
