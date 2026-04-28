import json
import argparse
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
# from geracl import GeraclHF
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification
from pathlib import Path


DATASET_PATH = "../../data/sets_to_learn/dataset_hier.json"
GERACL_PATH  = "../../models/GeRaCl-USER2-base"
OUTPUT_DIR   = "../../models/GeRaCl-finetuned721"

EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 256
N_NEGATIVES_PER_SAMPLE = 5

def set_seed(seed=666):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_cls_accuracy(records, model, tokenizer, level, device):
    model.eval()
    correct = 0
    total = len(records)

    with torch.no_grad():
        for r in records:
            parts = r[f"true_l{level}"].split(" ", 1)
            true_code = parts[0]
            candidates = r[f"candidates_l{level}"]

            if not candidates:
                total -= 1
                continue

            scores = []
            for c in candidates:
                enc = tokenizer(
                    r["summary"], c["name"],
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
    def __init__(self, records, level: int, tokenizer, max_len: int):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        lvl = f"l{level}"

        for r in records:
            summary    = r["summary"]
            candidates = r[f"candidates_{lvl}"]

            parts = r[f"true_{lvl}"].split(" ", 1)
            true_code = parts[0]
            true_name = parts[1] if len(parts) > 1 else next((c["name"] for c in candidates if c["code"] == true_code), None)
            if not true_name:
                continue

            self.samples.append((summary, true_name, 0))

            negs = [c["name"] for c in candidates if c["code"] != true_code]
            negs_sample = random.sample(negs, min(N_NEGATIVES_PER_SAMPLE, len(negs)))
            for neg in negs_sample:
                self.samples.append((summary, neg, 1))

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
    print(f"Обучаем GeRaCl-L{level}")

    set_seed()

    with open(DATASET_PATH, encoding="utf-8") as f:
        records = json.load(f)
        random.shuffle(records)
        n = len(records)
        train_rec = records[:int(n * 0.8)]
        val_rec = records[int(n * 0.8):int(n * 0.9)]
        test_rec = records[int(n * 0.9):]

    tokenizer = AutoTokenizer.from_pretrained(GERACL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        "deepvk/USER2-base", num_labels=2
    ).to("cuda")

    dataset = NLIDataset(train_rec, level, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(dataloader),
        num_training_steps=len(dataloader) * EPOCHS,
    )

    model.train()

    best_val_acc = 0
    best_val_cls_acc = 0.0
    val_dataset = NLIDataset(val_rec, level, tokenizer, MAX_LEN)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
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

        # model.eval()
        # val_loss, val_correct, val_total = 0, 0, 0
        # with torch.no_grad():
        #     for batch in val_dataloader:
        #         input_ids = batch["input_ids"].to("cuda")
        #         attention_mask = batch["attention_mask"].to("cuda")
        #         labels = batch["label"].to("cuda")
        #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        #         val_loss += outputs.loss.item()
        #         preds = outputs.logits.argmax(dim=-1)
        #         val_correct += (preds == labels).sum().item()
        #         val_total += len(labels)
        # val_acc = val_correct / val_total * 100
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     model.save_pretrained(str(out_path))
        #     tokenizer.save_pretrained(str(out_path))
        #     print(f"  ✓ Сохранена лучшая модель (val_acc={val_acc:.1f}%)")

        val_cls_acc = compute_cls_accuracy(val_rec, model, tokenizer, level, "cuda")
        print(f"  Val Cls Acc: {val_cls_acc:.4f}")

        if val_cls_acc > best_val_cls_acc:
            best_val_cls_acc = val_cls_acc
            model.save_pretrained(str(out_path))
            tokenizer.save_pretrained(str(out_path))
            print(f"  ✓ Сохранена лучшая модель (val_cls_acc={val_cls_acc:.4f})")

        # print(f"  Val     | loss={val_loss / len(val_dataloader):.4f} | acc={val_acc:.1f}%")
        model.train()

    # print("\n--- Test ---")
    # test_dataset = NLIDataset(test_rec, level, tokenizer, MAX_LEN)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    # model.eval()
    # test_correct, test_total = 0, 0
    # with torch.no_grad():
    #     for batch in test_dataloader:
    #         input_ids = batch["input_ids"].to("cuda")
    #         attention_mask = batch["attention_mask"].to("cuda")
    #         labels = batch["label"].to("cuda")
    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #         preds = outputs.logits.argmax(dim=-1)
    #         test_correct += (preds == labels).sum().item()
    #         test_total += len(labels)
    # print(f"  Test acc: {test_correct / test_total * 100:.1f}%")

    model = AutoModelForSequenceClassification.from_pretrained(str(out_path)).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(str(out_path))

    test_cls_acc = compute_cls_accuracy(test_rec, model, tokenizer, level, "cuda")
    print(f"  Test Classification Acc: {test_cls_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[2, 3, 4], required=True)
    args = parser.parse_args()
    train(args.level)
