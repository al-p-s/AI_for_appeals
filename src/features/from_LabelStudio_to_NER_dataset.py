import json
import re
from typing import List, Dict, Tuple
from collections import Counter


class LabelStudioToBIOConverter:
    def __init__(self, labelstudio_export_file):
        with open(labelstudio_export_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.labels = [
            "O",

            "B-LAST_NAME", "I-LAST_NAME",
            "B-FIRST_NAME", "I-FIRST_NAME",
            "B-MIDDLE_NAME", "I-MIDDLE_NAME",

            "B-PERSONAL_EMAIL", "I-PERSONAL_EMAIL",
            "B-GOV_EMAIL", "I-GOV_EMAIL",

            "B-PHONE_NUMBER", "I-PHONE_NUMBER",

            "B-ADDRESS", "I-ADDRESS",
            "B-POSTAL_CODE", "I-POSTAL_CODE",
            "B-REGION", "I-REGION",
            "B-CITY", "I-CITY",
            "B-STREET", "I-STREET",
            "B-HOUSE", "I-HOUSE",
            "B-ROOM", "I-ROOM",
            "B-DATE", "I-DATE"
        ]

        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def tokenize_with_positions(self, text: str) -> List[Tuple[str, int, int]]:
        pattern = r'\w+@\w+\.\w+|\+?[\d\-\(\)\s]{7,}|\w+|[^\w\s]'
        tokens = []
        for match in re.finditer(pattern, text):
            tokens.append((match.group(), match.start(), match.end()))
        return tokens

    def find_token_indices(self, tokens: List[Tuple], entity_start: int, entity_end: int) -> List[int]:
        indices = []
        for i, (_, token_start, token_end) in enumerate(tokens):
            if token_start < entity_end and token_end > entity_start:
                indices.append(i)
        return indices

    def convert_annotation_to_bio(self, task_id: str, text: str, annotation_result: List[Dict]) -> Dict:
        tokens_with_pos = self.tokenize_with_positions(text)
        tokens = [t[0] for t in tokens_with_pos]
        bio_tags = ["O"] * len(tokens)

        entities = sorted(
            [e for e in annotation_result if e['value']['labels'][0] != "MAIN_TEXT"],
            key=lambda x: x['value']['start']
        )

        for entity in entities:
            label = entity['value']['labels'][0]
            start = entity['value']['start']
            end = entity['value']['end']

            # Пропускаем метки, которых нет в нашем словаре
            if f"B-{label}" not in self.label2id:
                continue

            token_indices = self.find_token_indices(tokens_with_pos, start, end)

            if not token_indices:
                continue

            if any(bio_tags[idx] != "O" for idx in token_indices):
                continue

            bio_tags[token_indices[0]] = f"B-{label}"
            for idx in token_indices[1:]:
                bio_tags[idx] = f"I-{label}"

        ner_tag_ids = [self.label2id[tag] for tag in bio_tags]

        return {
            "id": str(task_id),
            "tokens": tokens,
            "ner_tags": ner_tag_ids,
            "text": text
        }

    def convert_all(self):
        converted = []
        stats = Counter()

        for task in self.data:
            if 'annotations' not in task or not task['annotations']:
                continue

            task_id = task.get("id")
            annotation = task['annotations'][0]
            text = task['data']['text']
            result = annotation.get('result', [])

            converted_example = self.convert_annotation_to_bio(task_id, text, result)
            converted.append(converted_example)

            for tag_id in converted_example['ner_tags']:
                stats[self.id2label[tag_id]] += 1

        return converted, stats

    def save_as_json(self, output_file, examples):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)


input_file = "../../data/NER_relearn2_0/raw_dataset.json"
output_file = "../../data/NER_relearn2_0/dataset.json"

converter = LabelStudioToBIOConverter(input_file)
examples, stats = converter.convert_all()

converter.save_as_json(output_file, examples)