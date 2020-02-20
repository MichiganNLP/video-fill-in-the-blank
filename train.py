import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import ActivityNetCaptionDataset

import numpy as np
import h5py
import json
import math
import nltk
import pickle

from baseline_BOW_VF import baseline_BOW_VF

print("Very begin")

folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"

videoFeatures = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
trainTextFile = f"{folder}/train.json"
valTextFile = f"{folder}/val_1.json"
# trainTextFile = 'train.json'
# valTextFile = 'val.json'

with open('word_dict.pkl', 'rb') as f:
    word_dict = pickle.load(f)

def train(data, max_epoch, model, optimizer, criterion):
    
    model.train()
    running_loss = 0
    for epoch in range(max_epoch):
        for n, batch in enumerate(data):
            optimizer.zero_grad()
            text_feature, video_feature, label = batch

            #use GPU
            text_feature = text_feature.cuda()
            video_feature = video_feature.cuda()
            label = label.cuda()

            output = model(text_feature, video_feature)
            # output = output.squeeze(dim=1)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if n%50 == 0 and n != 0:
                print("Epoch {}, batch {}: loss = {}".format(epoch, n, running_loss/50))
                running_loss = 0
    return model

def evaluation(test_data, model):
    model.eval()
    correct = 0
    total_num = 0

    for data in test_data:
        text_feature, video_feature, label = data

        text_feature = text_feature.cuda()
        video_feature = video_feature.cuda()
        label = label.cuda()

        output = model(text_feature, video_feature)
        batch_size = output.shape[0]
        for i in range(batch_size):
            if torch.argmax(output[i]).item() == label[i].item():
                correct += 1
            total_num += 1

    acc = correct / total_num
    print(acc)

print("start")

trainDataset = ActivityNetCaptionDataset(trainTextFile, videoFeatures, word_dict)
trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True, num_workers=4)
print("successfully load train")

valDataset = ActivityNetCaptionDataset(valTextFile, videoFeatures, word_dict, isTrain=False)
valLoader = DataLoader(valDataset, batch_size=16, shuffle=True, num_workers=4)
print("successfully load val")

model = baseline_BOW_VF(1000, 500, len(word_dict)).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
max_epoch = 20

model = train(trainLoader, max_epoch, model, optimizer, criterion)
evaluation(valLoader, model)


