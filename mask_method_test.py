import json
import csv
import torch
from transformers import BertForMaskedLM, BertTokenizer
import torch.nn.functional as F

JSON_FILE = "/scratch/mihalcea_root/mihalcea1/shared_data/VATEX/captions/vatex_training_v1.0.json"
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

data_out = []
for i in range(10):
    caption = data[i]
    sentence = caption["enCap"][0] # just pick the first one
    sentence_ids = tokenizer.encode(sentence)
    prob_min = (1, 0)
    prob_distance = (0, 0)
    prob_ratio = (0, 0)
    prob_rank = (0, 0)
    for i in range(1, len(sentence_ids) - 1):
        tmp_sentence = sentence_ids[:]
        label = tmp_sentence[i]
        tmp_sentence[i] = 103 # 103 is the id for [MASK]
        masked_lm_labels = torch.ones(1, len(tmp_sentence), dtype=torch.long) * -100
        masked_lm_labels[0,i] = label
        out = model(input_ids = torch.tensor(tmp_sentence).view(1,-1), masked_lm_labels = masked_lm_labels)
        probs = F.softmax(out[1][0][i], dim=0)
        prob = probs[label]
        _, sorted_index = torch.sort(probs, descending=True)
        
        rank = 0
        for r in range(sorted_index.shape[0]):
            if sorted_index[r] == label:
                rank = r
                break
        
        model_predict_word = tokenizer.convert_ids_to_tokens(torch.argmax(probs).item())
        max_prob = torch.max(probs).item()

        if prob_min[0] > prob:
            prob_min = (probs[label], label, prob, model_predict_word, max_prob)

        if prob_distance[0] < max_prob - prob:
            prob_distance = (max_prob - prob, label, prob, model_predict_word, max_prob)

        if prob_ratio[0] < prob / max_prob:
            prob_ratio = (prob / max_prob, label, prob, model_predict_word, max_prob)

        if prob_rank[0] < rank:
            prob_rank = (rank, label, prob, model_predict_word, max_prob)
    
    mask_based_on_prob = tokenizer.convert_ids_to_tokens(prob_min[1])
    mask_based_on_prob_distance = tokenizer.convert_ids_to_tokens(prob_distance[1])
    mask_based_on_prob_ratio = tokenizer.convert_ids_to_tokens(prob_ratio[1])
    mask_based_on_rank = tokenizer.convert_ids_to_tokens(prob_rank[1])

    data_out.append([sentence, mask_based_on_prob, *prob_min[2:],
                    mask_based_on_prob_distance, *prob_distance[2:],
                    mask_based_on_prob_ratio, *prob_ratio[2:],
                    mask_based_on_rank, *prob_rank[2:]])

with open("mask_method_test.csv", "w") as f:
    field_names = ["sentence", "lowest prob", "model_pred", "max_prob", "max prob distance", "model_pred", "max_prob", "max prob ratio", "model_pred", "max_prob", "max_prob_rank", "model_pred", "max_prob",]
    writer = csv.writer(f,delimiter=",")
    writer.writerow(field_names)
    for row in data_out:
        writer.writerow(row)
