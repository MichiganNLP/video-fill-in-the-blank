import json
import torch
from transformers import BertForMaskedLM, BertTokenizer
import torch.nn.functional as F

JSON_FILE = "/scratch/mihalcea_root/mihalcea1/shared_data/VATEX/captions/vatex_training_v1.0.json"
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

data_out = []
for caption in data:
    sentence = caption["enCap"][0] # just pick the first one
    sentence_ids = tokenizer.encode(sentence)
    prob_min = (1, 0)
    prob_distance = (0, 0)
    prob_ratio = (0, 0)
    for i in range(1, len(sentence_ids) - 1):
        tmp_sentence = sentence_ids[:]
        label = tmp_sentence[i]
        tmp_sentence[i] = 103 # 103 is the id for [MASK]
        masked_lm_labels = torch.ones(1, len(tmp_sentence), dtype=torch.long) * -100
        masked_lm_labels[0,i] = label
        out = model(input_ids = torch.tensor(tmp_sentence).view(1,-1), masked_lm_labels = masked_lm_labels)
        probs = F.softmax(out[1][0][i], dim=0)
        prob = probs[label]

        if prob_min[0] > prob:
            prob_min = (probs[label], label)
        
        max_prob = torch.max(probs).item()

        if prob_distance[0] < max_prob - prob:
            prob_distance = (max_prob - prob, label)

        if prob_ratio[0] < prob / max_prob:
            prob_ratio = (prob / max_prob, label)
    
    mask_based_on_prob = tokenizer.convert_ids_to_tokens(prob_min[1])
    mask_based_on_prob_distance = tokenizer.convert_ids_to_tokens(prob_distance[1])
    mask_based_on_prob_ratio = tokenizer.convert_ids_to_tokens(prob_ratio[1])
    
    data_out.append(sentence, mask_based_on_prob, mask_based_on_prob_distance, mask_based_on_prob_ratio)

pass