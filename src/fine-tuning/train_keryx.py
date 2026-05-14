import json
import argparse
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification
from pathlib import Path


DATASET_PATH = "../../data/sets_to_learn/dataset_hier720_v3.json"
CATS2_PATH = "../../data/classifier/cats2.json"
USER2_PATH = "../../models/USER2-base"
OUTPUT_DIR   = "../../models/KERYX_720p2"

BATCH_SIZE = 16
MAX_LEN = 256

LEVEL_CONFIG = {
    2: {"epochs": 5, "lr": 2e-5, "n_neg": 20},
    3: {"epochs": 5, "lr": 2e-5, "n_neg": 15},
    4: {"epochs": 4, "lr": 5e-6, "n_neg": 7},
}

def set_seed(seed=666):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_cls_accuracy(records, model, tokenizer, level, device, l2_candidates=None):
    model.eval()
    correct = 0
    total = len(records)

    with torch.no_grad():
        for r in records:
            parts = r[f"true_l{level}"].split(" ", 1)
            true_code = parts[0]

            if level == 2:
                candidates = l2_candidates
            else:
                candidates = r.get(f"candidates_l{level}")

            if not candidates:
                total -= 1
                continue

            scores = []
            if level == 2:
                prefix = ""
            elif level == 3:
                l2_parts = r["true_l2"].split(" ", 1)
                prefix = (l2_parts[1] if len(l2_parts) > 1 else l2_parts[0]) + " → "
            else:
                l2_parts = r["true_l2"].split(" ", 1)
                l2_name = l2_parts[1] if len(l2_parts) > 1 else l2_parts[0]
                l3_parts = r["true_l3"].split(" ", 1)
                l3_name = l3_parts[1] if len(l3_parts) > 1 else l3_parts[0]
                prefix = f"{l2_name} → {l3_name} → "

            for c in candidates:
                enc = tokenizer(
                    r["summary"], prefix + c["name"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LEN
                ).to(device)
                logits = model(**enc).logits
                scores.append(logits[0, 0].item())

            pred_code = candidates[scores.index(max(scores))]["code"]
            if pred_code == true_code:
                correct += 1

    return correct / total if total > 0 else 0.0

class NLIDataset(Dataset):
    def __init__(self, records, level: int, tokenizer, max_len: int, l2_candidates=None, n_neg=15):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        lvl = f"l{level}"

        for r in records:
            summary    = r["summary"]
            if level == 2:
                candidates = l2_candidates
                prefix = ""
            else:
                candidates = r.get(f"candidates_{lvl}")
                if level == 3:
                    l2_parts = r["true_l2"].split(" ", 1)
                    prefix = (l2_parts[1] if len(l2_parts) > 1 else l2_parts[0]) + " → "
                else:
                    l2_parts = r["true_l2"].split(" ", 1)
                    l2_name = l2_parts[1] if len(l2_parts) > 1 else l2_parts[0]
                    l3_parts = r["true_l3"].split(" ", 1)
                    l3_name = l3_parts[1] if len(l3_parts) > 1 else l3_parts[0]
                    prefix = f"{l2_name} → {l3_name} → "

            parts = r[f"true_{lvl}"].split(" ", 1)
            true_code = parts[0]
            true_name = parts[1] if len(parts) > 1 else next((c["name"] for c in candidates if c["code"] == true_code), None)
            if not true_name:
                continue

            self.samples.append((summary, prefix + true_name, 0))

            negs = [c["name"] for c in candidates if c["code"] != true_code]
            negs_sample = random.sample(negs, min(n_neg, len(negs)))
            for neg in negs_sample:
                self.samples.append((summary, prefix + neg, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        premise, hypothesis, label = self.samples[idx]
        enc = self.tokenizer(
            premise, hypothesis,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

def train(level: int):
    print(f"Обучаем KERYX-L{level}")

    set_seed()
    cfg = LEVEL_CONFIG[level]
    EPOCHS = cfg["epochs"]
    LR = cfg["lr"]

    with open(DATASET_PATH, encoding="utf-8") as f:
        records = json.load(f)
        random.shuffle(records)
        n = len(records)
        train_rec = records[:int(n * 0.8)]
        val_rec = records[int(n * 0.8):int(n * 0.9)]
        test_rec = records[int(n * 0.9):]

    with open(CATS2_PATH, encoding="utf-8") as f:
        l2_candidates = json.load(f)["categories"]

    tokenizer = AutoTokenizer.from_pretrained(USER2_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        USER2_PATH, num_labels=2
    ).to("cuda")

    dataset = NLIDataset(train_rec, level, tokenizer, MAX_LEN, l2_candidates, n_neg=cfg["n_neg"])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(dataloader) // 2,
        num_training_steps=len(dataloader) * EPOCHS,
    )

    model.train()

    best_val_cls_acc = 0.0
    out_path = Path(OUTPUT_DIR) / f"L{level}"
    out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["label"].to("cuda")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        acc = correct / total * 100
        print(f"  Epoch {epoch+1}/{EPOCHS} | loss={total_loss/len(dataloader):.4f} | acc={acc:.1f}%")

        val_cls_acc = compute_cls_accuracy(val_rec, model, tokenizer, level, "cuda", l2_candidates)
        print(f"  Val Cls Acc: {val_cls_acc:.4f}")

        if val_cls_acc > best_val_cls_acc:
            best_val_cls_acc = val_cls_acc
            model.save_pretrained(str(out_path))
            tokenizer.save_pretrained(str(out_path))
            print(f"  ✓ Сохранена лучшая модель (val_cls_acc={val_cls_acc:.4f})")

        model.train()

    model = AutoModelForSequenceClassification.from_pretrained(str(out_path)).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(str(out_path))

    test_cls_acc = compute_cls_accuracy(test_rec, model, tokenizer, level, "cuda", l2_candidates)
    print(f"  Test Classification Acc: {test_cls_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[2, 3, 4], required=True)
    args = parser.parse_args()
    train(args.level)
