import json
import torch
from transformers import BertModel, BertTokenizer

JSON_FILE = "/scratch/mihalcea_root/mihalcea1/shared_data/VATEX/captions/vatex_training_v1.0.json"
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

for caption in data:
    sentence = caption["enCap"][0] # just pick the first one
    bert_out = model(**tokenizer(sentence, return_tensors="pt"))