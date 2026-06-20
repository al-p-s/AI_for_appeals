import json

with open('../../../data/OCR_improving/appeals121_20_06_TEST.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for appeal in data['appeals']:
    print(appeal['file_name'])
    print(appeal['text'])
    print('-' * 50)
