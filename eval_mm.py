import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader_mulitmodal import ActivityNetCaptionDataset
from multi_modal_model import multi_modal_model
from transformers import BertTokenizer, BertForMaskedLM, AdamW
import h5py

def batchPadding(batch):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    batch_size = len(batch)

    textFeatures = []
    videoFeatures = []
    labels = []
    mask_positions = []
    lm_labels = []

    max_text_len = 0
    max_video_len = 0
    for i in range(batch_size):
        data = batch[i]
        text = torch.tensor(data[0])
        video = data[1]
        labels.append(data[2])
        mask_positions.append(data[3])
        
        textFeatures.append(text)
        videoFeatures.append(video)

        total_text_len = len(text)
        total_video_len = video.shape[0]
        if total_text_len > max_text_len:
            max_text_len = total_text_len
        if total_video_len > max_video_len:
            max_video_len = total_video_len
    
    text_tensor = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    video_tensor = torch.zeros(batch_size, max_video_len, video.shape[1], dtype=torch.float)

    segments_tensor = torch.cat([torch.zeros(batch_size, max_text_len, dtype=torch.long), torch.ones(batch_size, max_video_len, dtype=torch.long)], dim=1)
    attention_mask = torch.zeros(batch_size, max_text_len + max_video_len)
    masked_lm_labels = torch.ones(batch_size, max_text_len + max_video_len, dtype=torch.long) * (-100)
    

    for i in range(batch_size):
        text = textFeatures[i]
        video = videoFeatures[i]
        text_len = len(text)
        video_len = video.shape[0]

        text_tensor[i, :text_len-1] = text[:-1]
        text_tensor[i, -1] = text[-1]

        video_tensor[i, :video_len] = video

        attention_mask[i, :text_len-1] = 1
        attention_mask[i, max_text_len-1:max_text_len+video_len] = 1

        masked_lm_labels[i, mask_positions[i]] = tokenizer.convert_tokens_to_ids(labels[i])


    return (text_tensor, video_tensor, attention_mask, segments_tensor, labels, mask_positions, masked_lm_labels)

PATH = 'checkpoints/Checkpoint'
folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"

videoFeatures = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
valTextFile = f"{folder}/val_1.json"

bertModel = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=False)
embedding_size = 768
batch_size = 16

model = multi_modal_model(bertModel, 500, embedding_size)

model.load_state_dict(torch.load(PATH))
if torch.cuda.is_available():
    model = model.cuda()
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
    
    output = model(textFeatures, videoFeatures, attention_mask, segment_mask, mask_lm_labels)             
    batch_size = textFeatures.shape[0]
    score = output[1]
    predicted_index = torch.argmax(score[list(range(batch_size)), mask_positions], dim=1)

    top5=score[list(range(batch_size)), mask_positions].topk(5, dim=1)[1]
    for i in range(batch_size):
        print(''.join(tokenizer.convert_ids_to_tokens(textFeatures[i])))
        print(tokenizer.convert_ids_to_tokens(list(top5[i])))
        print(labels[i])
        print()

    out_text = tokenizer.convert_ids_to_tokens(predicted_index.tolist())
    total_num += batch_size
    for i in range(batch_size):
        if labels[i] == out_text[i]:
            correct += 1
acc = correct / total_num
print(acc)