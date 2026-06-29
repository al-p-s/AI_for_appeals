import json

with open('../../data/NER_relearn2_0/dataset.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

with open('../../data/NER_relearn2_0/train_dataset.json', 'w', encoding='utf-8') as f1:
    json.dump(data[:480], f1, ensure_ascii=False)

with open('../../data/NER_relearn2_0/validation_dataset.json', 'w', encoding='utf-8') as f2:
    json.dump(data[480:540], f2, ensure_ascii=False)

with open('../../data/NER_relearn2_0/test_dataset.json', 'w', encoding='utf-8') as f3:
    json.dump(data[540:], f3, ensure_ascii=False)