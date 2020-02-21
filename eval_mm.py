import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader_mulitmodal import ActivityNetCaptionDataset
from multi_modal_model import multi_modal_model
from transformers import BertTokenizer, BertForMaskedLM, AdamW
import h5py

PATH = 'Checkpoint'
folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"

videoFeatures = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
valTextFile = f"{folder}/val_1.json"

bertModel = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=False)
embedding_size = 768
batch_size = 16

model = multi_modal_model(bertModel, 500, embedding_size)

model.load_state_dict(torch.load(PATH))

model.eval()

valDataset = ActivityNetCaptionDataset(valTextFile, videoFeatures, isTrain=False)
val_dataLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True, collate_fn=batchPadding)

correct = 0
total_num = 0
for batch in val_dataLoader:
    textFeatures, videoFeatures, attention_mask, segment_mask, labels, mask_positions, mask_lm_labels = batch
    if torch.cuda.is_available():
        textFeatures = textFeatures.cuda()
        videoFeatures = videoFeatures.cuda()
        attention_mask = attention_mask.cuda()
        segment_mask = segment_mask.cuda()
        mask_lm_labels = mask_lm_labels.cuda()
        labels = labels.cuda()          
    
    output = model(textFeatures, videoFeatures, attention_mask, segment_mask, mask_lm_labels)             
    batch_size = textFeatures.shape[0]
    score = output[1]
    predicted_index = torch.argmax(score[list(range(batch_size)), mask_positions], dim=1)
    out_text = tokenizer.convert_ids_to_tokens(predicted_index.tolist())
    total_num += batch_size
    for i in range(batch_size):
        if labels[i] == out_text[i]:
            correct += 1
acc = correct / total_num
print(acc)