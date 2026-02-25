import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_path = r"C:/Users/USER/PycharmProjects/AI_for_appeals/models/NER_RuBert"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
model.eval()

def analyze_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [model.config.id2label[p.item()] for p in predictions[0]]

    for token, label in zip(tokens, labels):
        if token not in ['[CLS]', '[SEP]']:
            print(f"{token:15} -> {label}")

text = "Зиганьшина Надежда Алексеевна"
print(text)
analyze_text(text)

print(model.config.id2label)