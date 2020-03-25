import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader_multimodal import ActivityNetCaptionDataset
from multi_modal_model import multi_modal_model
from transformers import BertConfig, BertTokenizer, BertForMaskedLM, AdamW


def batchPadding(batch):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    batch_size = len(batch)

    textFeatures = []
    videoFeatures = []
    labels = []
    mask_positions = []
    lm_labels = []
    keys = []

    max_text_len = 0
    max_video_len = 0
    for i in range(batch_size):
        data = batch[i]
        text = torch.tensor(data[0])
        labels.append(data[2])
        mask_positions.append(data[3])
        keys.append(data[4])
        
        textFeatures.append(text)

        total_text_len = len(text)
        if total_text_len > max_text_len:
            max_text_len = total_text_len

    
    text_tensor = torch.zeros(batch_size, max_text_len, dtype=torch.long)

    segments_tensor = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_text_len)
    masked_lm_labels = torch.ones(batch_size, max_text_len, dtype=torch.long) * (-100)
    position_embedding = torch.arange(max_text_len, dtype=torch.long)
    position_embedding = position_embedding.view(1,-1).repeat(batch_size, 1)

    for i in range(batch_size):
        text = textFeatures[i]
        text_len = len(text)

        text_tensor[i, :text_len] = text

        attention_mask[i, :text_len] = 1

        masked_lm_labels[i, mask_positions[i]] = tokenizer.convert_tokens_to_ids(labels[i])


    return (text_tensor, attention_mask, segments_tensor, labels, mask_positions, masked_lm_labels, position_embedding, keys)

PATH = 'Checkpoint_textonly'
folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"

valTextFile = f"{folder}/val.pkl"

config = BertConfig()

model = BertForMaskedLM(config)
batch_size = 16

model.load_state_dict(torch.load(PATH))
if torch.cuda.is_available():
    model = model.cuda()
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

valDataset = ActivityNetCaptionDataset(valTextFile)
val_dataLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True, collate_fn=batchPadding, num_workers=4)

correct = 0
total_num = 0
for batch in val_dataLoader:
    textFeatures, attention_mask, segment_mask, labels, mask_positions, mask_lm_labels, position_embedding, key = batch
    if torch.cuda.is_available():
        textFeatures = textFeatures.cuda()
        attention_mask = attention_mask.cuda()
        segment_mask = segment_mask.cuda()
        mask_lm_labels = mask_lm_labels.cuda()
        position_embedding = position_embedding.cuda()   
    
    output = model(textFeatures, attention_mask = attention_mask, token_type_ids=segment_mask, masked_lm_labels=mask_lm_labels,position_ids=position_embedding)            
    batch_size = textFeatures.shape[0]
    score = output[1]
    predicted_index = torch.argmax(score[list(range(batch_size)), mask_positions], dim=1)

    top5=score[list(range(batch_size)), mask_positions].topk(5, dim=1)[1]
    with open("eval_scratch", 'a') as f:
        for i in range(batch_size):
            f.write(key[i])
            f.write('\n')
            f.write(' '.join(tokenizer.convert_ids_to_tokens(textFeatures[i])))
            f.write('\n')
            f.write(' '.join(tokenizer.convert_ids_to_tokens(list(top5[i]))))
            f.write('\n')
            f.write(labels[i])
            f.write('\n\n')

    out_text = tokenizer.convert_ids_to_tokens(predicted_index.tolist())
    total_num += batch_size
    for i in range(batch_size):
        if labels[i] == out_text[i]:
            correct += 1
acc = correct / total_num
print(acc)
