import json
import logging
import torch
import random
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from pathlib import Path

# CONFIG

FIELD_NAME = "RegistrationPlaceId"
N_NEG = 15 # for StatusId, RegistrationPlaceId, DeliveryTypeId, PetitionerCategory
# N_NEG = 4 # for AppealKind, ConsiderationType, ItemID, PetitionerDistrict, RegistrationPlaceId

FIELDS_CONFIG_PATH = "../../data/classifier/category_fields.json"
with open(FIELDS_CONFIG_PATH, encoding="utf-8") as f:
    fields_config = json.load(f)
cfg = next(c for c in fields_config if c["field_name"] == FIELD_NAME)
FIELD_CANDIDATES = cfg["field_candidates"]

DATASET_PATH = "../../data/sets_to_learn/fields/target_fields_v3.json"
USER2_PATH = "../../models/USER2-base"
OUTPUT_DIR = f"../../models/KERYXes_for_fields/KERYX_field_{FIELD_NAME}"

EPOCHS = 5
LR = 2e-5
BATCH_SIZE = 16
MAX_LEN = 512

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_true_value(item: dict, field_name: str) -> str:
    val = item.get(field_name, "")
    if isinstance(val, int):
        val = str(val)
    val = val.strip()
    return val if val else "не определяется"


class FieldNLIDataset(Dataset):
    def __init__(self, records, tokenizer, max_len, n_neg):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for r in records:
            text = r.get("text", "").strip()
            if not text:
                continue

            true_value = get_true_value(r, FIELD_NAME)

            # Позитив
            self.samples.append((text, true_value, 0))

            # Негативы — все остальные кандидаты
            negs = [c for c in FIELD_CANDIDATES if c != true_value]
            negs_sample = random.sample(negs, min(n_neg, len(negs)))
            for neg in negs_sample:
                self.samples.append((text, neg, 1))

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


def compute_accuracy(records, model, tokenizer, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for r in records:
            text = r.get("text", "").strip()
            if not text:
                continue

            true_value = get_true_value(r, FIELD_NAME)
            total += 1

            scores = []
            for candidate in FIELD_CANDIDATES:
                enc = tokenizer(
                    text, candidate,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LEN,
                ).to(device)
                logits = model(**enc).logits
                scores.append(logits[0, 0].item())

            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            pred_value = FIELD_CANDIDATES[best_idx]

            if pred_value == true_value:
                correct += 1

    return correct / total if total > 0 else 0.0


def train():
    log_path = Path(f"logs/fields/train_{FIELD_NAME}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(str(log_path), encoding="utf-8"),
        ]
    )
    logger = logging.getLogger(__name__)

    def log(msg):
        print(msg)
        logger.info(msg)

    log(f"Training field classifier: {FIELD_NAME}")
    log(f"Candidates ({len(FIELD_CANDIDATES)}): {FIELD_CANDIDATES}")

    set_seed()

    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    records = data["appeals"]

    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[get_true_value(r, FIELD_NAME)].append(r)

    train_rec, val_rec = [], []
    for group in groups.values():
        random.shuffle(group)
        n_val = max(1, int(len(group) * 0.1))
        val_rec.extend(group[:n_val])
        train_rec.extend(group[n_val:])

    random.shuffle(train_rec)
    random.shuffle(val_rec)
    log(f"Train: {len(train_rec)} | Val: {len(val_rec)}")

    dist = Counter(get_true_value(r, FIELD_NAME) for r in train_rec)
    log(f"Class distribution (train): {dict(dist)}")

    tokenizer = AutoTokenizer.from_pretrained(USER2_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        USER2_PATH, num_labels=2
    ).to("cuda")

    dataset = FieldNLIDataset(train_rec, tokenizer, MAX_LEN, N_NEG)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(dataloader) // 2,
        num_training_steps=len(dataloader) * EPOCHS,
    )

    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
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

        train_acc = correct / total * 100
        log(f"  Epoch {epoch+1}/{EPOCHS} | loss={total_loss/len(dataloader):.4f} | acc={train_acc:.1f}%")

        val_acc = compute_accuracy(val_rec, model, tokenizer, "cuda")
        log(f"  Val accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(str(out_path))
            tokenizer.save_pretrained(str(out_path))
            log(f"  [UPD] Best model saved (acc={val_acc:.4f})")

    log(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    log(f"Model saved to: {out_path}")


if __name__ == "__main__":
    train()
