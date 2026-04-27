import json
import argparse
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from geracl import GeraclHF
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification
from pathlib import Path


DATASET_PATH = "../../data/sets_to_learn/dataset_hier.json"
GERACL_PATH  = "../../models/GeRaCl-USER2-base"
OUTPUT_DIR   = "../../models/GeRaCl-finetuned"

EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 256
NEG_PER_POS = 15 # сколько негативных примеров на один позитивный

class NLIDataset(Dataset):
    def __init__(self, records, level: int, tokenizer, max_len: int):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        lvl = f"l{level}"

        for r in records:
            summary    = r["summary"]
            true_code  = r[f"true_{lvl}"]
            candidates = r[f"candidates_{lvl}"]

            true_name = next((c["name"] for c in candidates if c["code"] == true_code), None)
            if not true_name:
                continue

            # Позитив: entailment = 0
            self.samples.append((summary, true_name, 0))

            # Негативы: contradiction = 2
            negs = [c["name"] for c in candidates if c["code"] != true_code]
            random.shuffle(negs)
            for neg in negs[:NEG_PER_POS]:
                self.samples.append((summary, neg, 2))

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
            # "token_type_ids": enc.get("token_type_ids", torch.zeros(self.max_len, dtype=torch.long)).squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

def train(level: int):
    print(f"Обучаем GeRaCl-L{level}")

    with open(DATASET_PATH, encoding="utf-8") as f:
        records = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(GERACL_PATH)
    # model = GeraclHF.from_pretrained(GERACL_PATH).to("cuda")
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     GERACL_PATH, num_labels=3
    # ).to("cuda")
    model = AutoModelForSequenceClassification.from_pretrained(
        "deepvk/USER2-base", num_labels=3
    ).to("cuda")

    dataset = NLIDataset(records, level, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(dataloader),
        num_training_steps=len(dataloader) * EPOCHS,
    )

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            # token_type_ids = batch["token_type_ids"].to("cuda")
            labels = batch["label"].to("cuda")

            # outputs = model(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     # token_type_ids=token_type_ids,
            #     labels=labels,
            # )

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

    out_path = Path(OUTPUT_DIR) / f"L{level}"
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"Сохранено → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[2, 3, 4], required=True)
    args = parser.parse_args()
    train(args.level)
