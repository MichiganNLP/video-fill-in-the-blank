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

def train(data, max_epoch, model, optimizer, PATH):
    
    model.train()
    running_loss = 0
    for epoch in range(max_epoch):

        for n, batch in enumerate(data):
            optimizer.zero_grad()
            textFeatures, videoFeatures, attention_mask, segment_mask, labels, mask_positions, mask_lm_labels = batch
            if torch.cuda.is_available():
                textFeatures = textFeatures.cuda()
                videoFeatures = videoFeatures.cuda()
                attention_mask = attention_mask.cuda()
                segment_mask = segment_mask.cuda()
                mask_lm_labels = mask_lm_labels.cuda()
            
            output = model(textFeatures, videoFeatures, attention_mask, segment_mask, mask_lm_labels)
            loss = output[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if n%50 == 0 and n != 0:
                print("Epoch {}, batch {}: loss = {}".format(epoch, n, running_loss/50))
                running_loss = 0
        torch.save(model.state_dict(), PATH)
    return model

def main():
    PATH = 'Checkpoint'
    folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"

    videoFeatures = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
    trainTextFile = f"{folder}/train.json"
    valTextFile = f"{folder}/val_1.json"

    bertModel = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=False)
    embedding_size = 768
    max_epoch = 20
    batch_size = 16
    lr = 0.0001

    model = multi_modal_model(bertModel, 500, embedding_size)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = AdamW(model.parameters(), lr=lr)  

    trainDataset = ActivityNetCaptionDataset(trainTextFile, videoFeatures)
    # valDataset = ActivityNetCaptionDataset(valTextFile, videoFeatures, isTrain=False)

    train_dataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, collate_fn=batchPadding)
    # val_dataLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True, collate_fn=batchPadding)

    train(train_dataLoader, max_epoch, model, optimizer, PATH)

if __name__ == "__main__":
    main()
