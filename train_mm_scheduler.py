import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader_multimodal import ActivityNetCaptionDataset
from multi_modal_model import multi_modal_model
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
import h5py
from utils import batchPadding

def train(data, max_epoch, model, optimizer, scheduler PATH):
    
    model.train()
    running_loss = 0
    for epoch in range(max_epoch):

        for n, batch in enumerate(data):
            optimizer.zero_grad()
            textFeatures, videoFeatures, attention_mask, segment_mask, labels, mask_positions, mask_lm_labels, position_embedding, _ = batch
            if torch.cuda.is_available():
                textFeatures = textFeatures.cuda()
                videoFeatures = videoFeatures.cuda()
                attention_mask = attention_mask.cuda()
                segment_mask = segment_mask.cuda()
                mask_lm_labels = mask_lm_labels.cuda()
                position_embedding = position_embedding.cuda()
            
            output = model(textFeatures, videoFeatures, attention_mask, segment_mask, mask_lm_labels,position_embedding)
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
    PATH = 'Checkpoint_scheduler'
    folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"

    # videoFeatures = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
    trainFile = f"{folder}/train.pkl"

    bertModel = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=False)
    embedding_size = 768
    max_epoch = 10
    batch_size = 16
    lr = 0.0001

    model = multi_modal_model(bertModel, 500, embedding_size)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = AdamW(model.parameters(), lr=lr)  

    trainDataset = ActivityNetCaptionDataset(trainFile)

    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * max_epoch, max_epoch)

    train_dataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, collate_fn=batchPadding, num_workers=8)

    train(train_dataLoader, max_epoch, model, optimizer, scheduler, PATH)

if __name__ == "__main__":
    main()
