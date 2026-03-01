import json
import re
from typing import List, Dict, Tuple
from collections import Counter

class LabelStudioToBIOLUConverter:
    def __init__(self, labelstudio_export_file):
        with open(labelstudio_export_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.all_labels = [
            "O",
            "B-LAST_NAME", "I-LAST_NAME", "L-LAST_NAME", "U-LAST_NAME",
            "B-FIRST_NAME", "I-FIRST_NAME", "L-FIRST_NAME", "U-FIRST_NAME",
            "B-MIDDLE_NAME", "I-MIDDLE_NAME", "L-MIDDLE_NAME", "U-MIDDLE_NAME",
            "B-PERSONAL_EMAIL", "I-PERSONAL_EMAIL", "L-PERSONAL_EMAIL", "U-PERSONAL_EMAIL",
            "B-GOV_EMAIL", "I-GOV_EMAIL", "L-GOV_EMAIL", "U-GOV_EMAIL",
            "B-MAIN_TEXT", "I-MAIN_TEXT", "L-MAIN_TEXT", "U-MAIN_TEXT"
        ]

    def tokenize_with_positions(self, text: str) -> List[Tuple[str, int, int]]:
        tokens = []
        pattern = r'\S+|[^\w\s]'

        for match in re.finditer(pattern, text):
            token = match.group()
            start = match.start()
            end = match.end()
            tokens.append((token, start, end))

        return tokens

    def find_token_indices(self, tokens: List[Tuple], entity_start: int, entity_end: int) -> List[int]:
        indices = []
        for i, (_, token_start, token_end) in enumerate(tokens):
            if token_start < entity_end and token_end > entity_start:
                indices.append(i)
        return indices

    def convert_annotation_to_biolu(self, text: str, annotation_result: List[Dict]) -> Dict:
        tokens_with_pos = self.tokenize_with_positions(text)
        tokens = [t[0] for t in tokens_with_pos]

        bio_tags = ["O"] * len(tokens)

        entities = sorted(annotation_result,
                          key=lambda x: x['value']['start'],
                          reverse=True)

        for entity in entities:
            label = entity['value']['labels'][0]
            start = entity['value']['start']
            end = entity['value']['end']

            token_indices = self.find_token_indices(tokens_with_pos, start, end)

            if not token_indices:
                print(f"Предупреждение: сущность '{entity['value']['text']}' не найдена в токенах")
                continue

            if len(token_indices) == 1:
                bio_tags[token_indices[0]] = f"U-{label}"
            else:
                bio_tags[token_indices[0]] = f"B-{label}"
                for idx in token_indices[1:-1]:
                    bio_tags[idx] = f"I-{label}"
                bio_tags[token_indices[-1]] = f"L-{label}"

        return {
            'text': text,
            'tokens': tokens,
            'ner_tags': bio_tags
        }

    def convert_all(self):
        converted = []
        stats = Counter()

        for task in self.data:
            if 'annotations' not in task or not task['annotations']:
                continue

            annotation = task['annotations'][0]
            text = task['data']['text']

            result = annotation.get('result', [])

            converted_example = self.convert_annotation_to_biolu(text, result)
            converted.append(converted_example)

            for tag in converted_example['ner_tags']:
                stats[tag] += 1

        return converted, stats

    def save_as_jsonl(self, output_file, examples):
        with open(output_file, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

input_file = "../../data/appeals_with_marks.json"

converter = LabelStudioToBIOLUConverter(input_file)

examples, stats = converter.convert_all()

converter.save_as_jsonl("../../data/training_data_biolu.jsonl", examples)
