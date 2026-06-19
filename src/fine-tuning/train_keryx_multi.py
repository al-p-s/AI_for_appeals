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


DATASET_PATH = "../../data/sets_to_learn/dataset_1223.json"
CATS2_PATH = "../../data/classifier/cats2.json"
USER2_PATH = "../../models/USER2-base"
OUTPUT_DIR = "../../models/KERYX_1223"

BATCH_SIZE = 16
MAX_LEN = 256

LEVEL_CONFIG = {
    2: {"epochs": 5, "lr": 2e-5, "n_neg": 20},
    3: {"epochs": 5, "lr": 2e-5, "n_neg": 15},
    4: {"epochs": 5, "lr": 1e-5, "n_neg": 12},
}

def set_seed(seed=666):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_cls_accuracy(records, model, tokenizer, level, device, name_by_l2_code, l2_candidates=None):
    model.eval()
    correct = 0
    total = 0
    jaccard_sum = 0.0
    partial_correct = 0

    with torch.no_grad():
        for r in records:
            if level == 2:
                candidates = l2_candidates
                if not candidates:
                    continue
                total += 1
                true_codes = {s.split(" ")[0] for s in r["true_l2"]}
                n = len(true_codes)
                scores = []
                for c in candidates:
                    enc = tokenizer(r["summary"], c["name"],
                                    return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
                    scores.append(model(**enc).logits[0, 0].item())
                top_n = {candidates[i]["code"] for i in sorted(range(len(scores)), key=lambda x: -scores[x])[:n]}
                if top_n == true_codes:
                    correct += 1
                jaccard_sum += len(top_n & true_codes) / len(top_n | true_codes)
                if top_n & true_codes:
                    partial_correct += 1

            elif level == 3:
                record_correct = True
                has_valid = False
                record_jaccard = 0.0
                record_jaccard_count = 0
                record_partial_count = 0
                record_partial_hit = 0
                for true_l2_str in r["true_l2"]:
                    l2_parts = true_l2_str.split(" ", 1)
                    l2_code = l2_parts[0]
                    l2_name = l2_parts[1] if len(l2_parts) > 1 else l2_parts[0]
                    prefix = l2_name + " → "
                    candidates = r.get("candidates_l3", {}).get(l2_code, [])
                    if not candidates:
                        continue
                    has_valid = True
                    true_codes = {
                        s.split(" ")[0] for s in r["true_l3"]
                        if s.split(" ")[0].startswith(".".join(l2_code.split(".")[:2]))
                    }
                    n = len(true_codes)
                    scores = []
                    for c in candidates:
                        enc = tokenizer(r["summary"], prefix + c["name"],
                                        return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
                        scores.append(model(**enc).logits[0, 0].item())
                    top_n = {candidates[i]["code"] for i in sorted(range(len(scores)), key=lambda x: -scores[x])[:n]}
                    if top_n != true_codes:
                        record_correct = False
                    record_partial_count += 1
                    if top_n & true_codes:
                        record_partial_hit += 1
                    if true_codes | top_n:
                        record_jaccard += len(top_n & true_codes) / len(top_n | true_codes)
                        record_jaccard_count += 1
                if has_valid:
                    total += 1
                    if record_correct:
                        correct += 1
                    if record_partial_count > 0:
                        partial_correct += record_partial_hit / record_partial_count
                    if record_jaccard_count:
                        jaccard_sum += record_jaccard / record_jaccard_count
                continue

            else:  # level == 4
                record_correct = True
                has_valid = False
                record_jaccard = 0.0
                record_jaccard_count = 0
                record_partial_count = 0
                record_partial_hit = 0
                for true_l3_str in r["true_l3"]:
                    l3_parts = true_l3_str.split(" ", 1)
                    l3_code = l3_parts[0]
                    l3_name = l3_parts[1] if len(l3_parts) > 1 else l3_parts[0]
                    l2_name = name_by_l2_code.get(".".join(l3_code.split(".")[:2]) + ".0000.0000", l3_code)
                    prefix = f"{l2_name} → {l3_name} → "
                    candidates = r.get("candidates_l4", {}).get(l3_code, [])
                    if not candidates:
                        continue
                    has_valid = True
                    true_codes = {
                        s.split(" ")[0] for s in r["true_l4"]
                        if s.split(" ")[0].startswith(".".join(l3_code.split(".")[:3]))
                    }
                    n = len(true_codes)
                    scores = []
                    for c in candidates:
                        enc = tokenizer(
                            r["summary"], prefix + c["name"],
                            return_tensors="pt", truncation=True, max_length=MAX_LEN
                        ).to(device)
                        logits = model(**enc).logits
                        scores.append(logits[0, 0].item())
                    top_n = {candidates[i]["code"] for i in sorted(range(len(scores)), key=lambda x: -scores[x])[:n]}
                    if top_n != true_codes:
                        record_correct = False
                    record_partial_count += 1
                    if top_n & true_codes:
                        record_partial_hit += 1
                    if true_codes | top_n:
                        record_jaccard += len(top_n & true_codes) / len(top_n | true_codes)
                        record_jaccard_count += 1
                if has_valid:
                    total += 1
                    if record_correct:
                        correct += 1
                    if record_partial_count > 0:
                        partial_correct += record_partial_hit / record_partial_count
                    if record_jaccard_count:
                        jaccard_sum += record_jaccard / record_jaccard_count
                continue

    return {
        "exact": correct / total if total > 0 else 0.0,
        "jaccard": jaccard_sum / total if total > 0 else 0.0,
        "partial": partial_correct / total if total > 0 else 0.0,
    }

class NLIDataset(Dataset):
    def __init__(self, records, level: int, tokenizer, max_len: int, l2_candidates=None, n_neg=15, name_by_l2_code=None):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for r in records:
            summary = r["summary"]
            if level == 2:
                candidates = l2_candidates
                true_codes_set = set()
                for true_str in r["true_l2"]:
                    parts = true_str.split(" ", 1)
                    code = parts[0]
                    name = parts[1] if len(parts) > 1 else (
                        next((c["name"] for c in candidates if c["code"] == code), None))
                    if not name:
                        continue
                    true_codes_set.add(code)
                    self.samples.append((summary, name, 0))
                negs = [c["name"] for c in candidates if c["code"] not in true_codes_set]
                negs_sample = random.sample(negs, min(n_neg, len(negs)))
                for neg in negs_sample:
                    self.samples.append((summary, neg, 1))

            elif level == 3:
                for true_l2_str in r["true_l2"]:
                    l2_parts = true_l2_str.split(" ", 1)
                    l2_code = l2_parts[0]
                    l2_name = l2_parts[1] if len(l2_parts) > 1 else l2_parts[0]
                    prefix = l2_name + " → "
                    candidates = r.get("candidates_l3", {}).get(l2_code, [])
                    if not candidates:
                        continue
                    true_codes_set = set()
                    for true_str in r["true_l3"]:
                        parts = true_str.split(" ", 1)
                        code = parts[0]
                        l2_prefix = ".".join(l2_code.split(".")[:2])
                        if not code.startswith(l2_prefix):
                            continue
                        name = parts[1] if len(parts) > 1 else next(
                            (c["name"] for c in candidates if c["code"] == code), None)
                        if not name:
                            continue
                        true_codes_set.add(code)
                        self.samples.append((summary, prefix + name, 0))
                    negs = [c["name"] for c in candidates if c["code"] not in true_codes_set]
                    negs_sample = random.sample(negs, min(n_neg, len(negs)))
                    for neg in negs_sample:
                        self.samples.append((summary, prefix + neg, 1))

            else:  # level == 4
                for true_l3_str in r["true_l3"]:
                    l3_parts = true_l3_str.split(" ", 1)
                    l3_code = l3_parts[0]
                    l3_name = l3_parts[1] if len(l3_parts) > 1 else l3_parts[0]
                    l2_code_from_l3 = ".".join(l3_code.split(".")[:2]) + ".0000.0000"
                    l2_name = name_by_l2_code.get(l2_code_from_l3, "")
                    prefix = f"{l2_name} → {l3_name} → "
                    candidates = r.get("candidates_l4", {}).get(l3_code, [])
                    if not candidates:
                        continue
                    true_codes_set = set()
                    for true_str in r["true_l4"]:
                        parts = true_str.split(" ", 1)
                        code = parts[0]
                        if not code.startswith(".".join(l3_code.split(".")[:3])):
                            continue
                        name = parts[1] if len(parts) > 1 else next(
                            (c["name"] for c in candidates if c["code"] == code), None)
                        if not name:
                            continue
                        true_codes_set.add(code)
                        self.samples.append((summary, prefix + name, 0))
                    negs = [c["name"] for c in candidates if c["code"] not in true_codes_set]
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
    print(f"Training KERYX-L{level}")

    set_seed()
    cfg = LEVEL_CONFIG[level]
    EPOCHS = cfg["epochs"]
    LR = cfg["lr"]

    with open(DATASET_PATH, encoding="utf-8") as f:
        records = json.load(f)
    random.seed(66)
    random.shuffle(records)
    n = len(records)
    train_rec = records[:int(n * 0.9)]
    val_rec = records[int(n * 0.9):]
    # train_rec = records[:int(n * 0.8)]
    # val_rec = records[int(n * 0.8):int(n * 0.9)]
    # test_rec = records[int(n * 0.9):]

    with open(CATS2_PATH, encoding="utf-8") as f:
        l2_candidates = json.load(f)["categories"]

    name_by_l2_code = {c["code"]: c["name"] for c in l2_candidates}
    tokenizer = AutoTokenizer.from_pretrained(USER2_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        USER2_PATH, num_labels=2
    ).to("cuda")

    dataset = NLIDataset(train_rec, level, tokenizer, MAX_LEN, l2_candidates, n_neg=cfg["n_neg"], name_by_l2_code=name_by_l2_code)
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

        val_metrics = compute_cls_accuracy(val_rec, model, tokenizer, level, "cuda", name_by_l2_code, l2_candidates)
        print(f"  Val: exact={val_metrics['exact']:.4f} jaccard={val_metrics['jaccard']:.4f} partial={val_metrics['partial']:.4f}")

        val_score = val_metrics['jaccard'] if level == 4 else val_metrics['exact']

        if val_score > best_val_cls_acc:
            best_val_cls_acc = val_score
            model.save_pretrained(str(out_path))
            tokenizer.save_pretrained(str(out_path))
            print(f"  [UPD] Best model saved (score={val_score:.4f})")

        model.train()

        # model = AutoModelForSequenceClassification.from_pretrained(str(out_path)).to("cuda")
        # tokenizer = AutoTokenizer.from_pretrained(str(out_path))
        #
        # test_metrics = compute_cls_accuracy(test_rec, model, tokenizer, level, "cuda", name_by_l2_code, l2_candidates)
        # print(f"  Test: exact={test_metrics['exact']:.4f} jaccard={test_metrics['jaccard']:.4f} partial={test_metrics['partial']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[2, 3, 4], required=True)
    args = parser.parse_args()
    train(args.level)
