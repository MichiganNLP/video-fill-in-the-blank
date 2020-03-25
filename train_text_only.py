import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader_multimodal import ActivityNetCaptionDataset
from multi_modal_model import multi_modal_model
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
import h5py

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

def train(data, max_epoch, model, optimizer, scheduler, PATH):
    
    model.train()
    running_loss = 0
    for epoch in range(max_epoch):

        for n, batch in enumerate(data):
            optimizer.zero_grad()
            textFeatures, attention_mask, segment_mask, labels, mask_positions, mask_lm_labels, position_embedding, _ = batch
            if torch.cuda.is_available():
                textFeatures = textFeatures.cuda()
                attention_mask = attention_mask.cuda()
                segment_mask = segment_mask.cuda()
                mask_lm_labels = mask_lm_labels.cuda()
                position_embedding = position_embedding.cuda()
            
            output = model(textFeatures, attention_mask = attention_mask, token_type_ids=segment_mask, masked_lm_labels=mask_lm_labels,position_ids=position_embedding)
            loss = output[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if n%50 == 0 and n != 0:
                print("Epoch {}, batch {}: loss = {}".format(epoch, n, running_loss/50))
                running_loss = 0
        scheduler.step()
        torch.save(model.state_dict(), PATH)
    return model

def main():
    PATH = 'Checkpoint_textonly'
    folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"

    # videoFeatures = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
    trainFile = f"{folder}/train.pkl"

    model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=False)
    max_epoch = 10
    batch_size = 16
    lr = 0.0001

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1)  

    trainDataset = ActivityNetCaptionDataset(trainFile)

    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * max_epoch, max_epoch)

    train_dataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, collate_fn=batchPadding, num_workers=8)

    train(train_dataLoader, max_epoch, model, optimizer, scheduler, PATH)

if __name__ == "__main__":
    main()
