import json

with open('../../data/training_data_hf.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

with open('../../data/train_dataset.json', 'w', encoding='utf-8') as f1:
    json.dump(data[:158], f1)

with open('../../data/validation_dataset.json', 'w', encoding='utf-8') as f2:
    json.dump(data[158:192], f2)

with open('../../data/test_dataset.json', 'w', encoding='utf-8') as f3:
    json.dump(data[192:], f3)