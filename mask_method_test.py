import json
import torch
from transformers import BertForMaskedLM, BertTokenizer

JSON_FILE = "/scratch/mihalcea_root/mihalcea1/shared_data/VATEX/captions/vatex_training_v1.0.json"
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

for caption in data:
    sentence = caption["enCap"][0] # just pick the first one
    sentence_ids = tokenizer.encode(sentence)
    for i in range(1, len(sentence_ids) - 1):
        tmp_sentence = sentence_ids[:]
        label = tmp_sentence[i]
        tmp_sentence[i] = 103 # 103 is the id for [MASK]
        masked_lm_labels = torch.ones(1, len(tmp_sentence), dtype=torch.long) * -100
        masked_lm_labels[i] = label
        out = model(input_ids = torch.tensor(tmp_sentence).view(1,-1), masked_lm_labels = masked_lm_labels)
