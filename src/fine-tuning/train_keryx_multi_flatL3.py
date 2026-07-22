import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification
from pathlib import Path


DATASET_PATH = "../../data/sets_to_learn/dataset_1340_G.json"
CATS3_PATH = "../../data/classifier/cats3.json"
USER2_PATH = "../../models/USER2-base"
OUTPUT_DIR = "../../models/KERYX_1340_flatL3"

BATCH_SIZE = 16
MAX_LEN = 256

# ~200+ кандидатов, конфиг аналогичен тому, что для L2 в каскадной версии
EPOCHS = 5
LR = 2e-5
N_NEG = 20


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_cls_accuracy(records, model, tokenizer, device, l3_candidates):
    model.eval()
    correct = 0
    total = 0
    jaccard_sum = 0.0
    partial_correct = 0

    with torch.no_grad():
        for r in records:
            total += 1
            true_codes = {s.split(" ")[0] for s in r["true_l3"]}
            n = len(true_codes)
            scores = []
            for c in l3_candidates:
                enc = tokenizer(r["summary"], c["name"],
                                return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
                scores.append(model(**enc).logits[0, 0].item())
            top_n = {l3_candidates[i]["code"] for i in sorted(range(len(scores)), key=lambda x: -scores[x])[:n]}
            if top_n == true_codes:
                correct += 1
            jaccard_sum += len(top_n & true_codes) / len(top_n | true_codes)
            if top_n & true_codes:
                partial_correct += 1

    return {
        "exact": correct / total if total > 0 else 0.0,
        "jaccard": jaccard_sum / total if total > 0 else 0.0,
        "partial": partial_correct / total if total > 0 else 0.0,
    }


class NLIDataset(Dataset):
    def __init__(self, records, tokenizer, max_len: int, l3_candidates, n_neg=20):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for r in records:
            summary = r["summary"]
            true_codes_set = set()
            for true_str in r["true_l3"]:
                parts = true_str.split(" ", 1)
                code = parts[0]
                name = parts[1] if len(parts) > 1 else (
                    next((c["name"] for c in l3_candidates if c["code"] == code), None))
                if not name:
                    continue
                true_codes_set.add(code)
                self.samples.append((summary, name, 0))
            negs = [c["name"] for c in l3_candidates if c["code"] not in true_codes_set]
            negs_sample = random.sample(negs, min(n_neg, len(negs)))
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


def train():
    print("Training KERYX-flatL3")

    set_seed()

    with open(DATASET_PATH, encoding="utf-8") as f:
        records = json.load(f)
    random.seed(42)
    random.shuffle(records)
    n = len(records)
    train_rec = records[:int(n * 0.9)]
    val_rec = records[int(n * 0.9):]

    with open(CATS3_PATH, encoding="utf-8") as f:
        l3_candidates = json.load(f)["categories"]

    tokenizer = AutoTokenizer.from_pretrained(USER2_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        USER2_PATH, num_labels=2
    ).to("cuda")

    dataset = NLIDataset(train_rec, tokenizer, MAX_LEN, l3_candidates, n_neg=N_NEG)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(dataloader) // 2,
        num_training_steps=len(dataloader) * EPOCHS,
    )

    model.train()

    best_val_cls_acc = 0.0
    out_path = Path(OUTPUT_DIR) / "L3"
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

        val_metrics = compute_cls_accuracy(val_rec, model, tokenizer, "cuda", l3_candidates)
        print(f"  Val: exact={val_metrics['exact']:.4f} jaccard={val_metrics['jaccard']:.4f} partial={val_metrics['partial']:.4f}")

        val_score = val_metrics['exact']

        if val_score > best_val_cls_acc:
            best_val_cls_acc = val_score
            model.save_pretrained(str(out_path))
            tokenizer.save_pretrained(str(out_path))
            print(f"  [UPD] Best model saved (score={val_score:.4f})")

        model.train()


if __name__ == "__main__":
    train()
