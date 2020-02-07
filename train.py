import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import ActivityNetCaptionDataset

import numpy as np
import h5py
import json
import math
import nltk
import random

from baseline_BOW_VF import baseline_BOW_VF

videoFeature = h5py.File("/scratch/mihalcea_root/mihalcea1/ruoyaow/ActivityNet_Captions/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
trainTextFile = "/scratch/mihalcea_root/mihalcea1/ruoyaow/ActivityNet_Captions/train.json"
valTextFile = "/scratch/mihalcea_root/mihalcea1/ruoyaow/ActivityNet_Captions/val_1.json"
# trainTextFile = 'train.json'
# valTextFile = 'val.json'

wordDict = {}
answerWordDict = {}
THRESHOLD = 500
wordID = 1

def getVideoFeatures(key, startFrame, endFrame):
    feature_h5 = videoFeature[key]['c3d_features']
    shape = feature_h5.shape
    feature_np = np.zeros(shape)
    feature_h5.read_direct(feature_np)
    feature = feature_np.mean(axis=0)
    return torch.FloatTensor(feature).view(1, -1)

def gen(text, parsed_sentence, isTrain):
    position = []
    for i in range(len(parsed_sentence)):
        if parsed_sentence[i][1] == 'JJ' or parsed_sentence[i][1] == 'NN':
            position.append(i)

    while len(position): 
        idx = random.sample(position, 1)[0]
        if text[idx] in answerWordDict and answerWordDict[text[idx]] >= THRESHOLD and isTrain:
            position.remove(idx)
            continue
        new_sentence = text[:]

        correct_word = new_sentence[idx]
        if isTrain:
            if correct_word in answerWordDict:
                answerWordDict[correct_word] += 1
            else:
                answerWordDict[correct_word] = 1

            for word in text:
                if word in wordDict:
                    wordDict[word]["freq"] += 1
                else:
                    wordDict[word] = {"id": wordID, "freq": 1}

        new_sentence[idx] = '[MASK]'
        return new_sentence, correct_word
    
    return ()

def getTextFeatures(textFile, isTrain=True):
    with open(textFile, 'r') as f:
        raw = json.load(f)
    
    data = []

    counter = 0
    for key in raw.keys():
        if isTrain:
            if counter > 1000:
                break
            else if counter > 100:
                break
        counter += 1
        total_events = len(raw[key]['sentences'])
        for i in range(total_events):
            start_frame = math.floor(raw[key]['timestamps'][i][0] * 2)
            end_frame = math.ceil(raw[key]['timestamps'][i][1] * 2)
            sentence = raw[key]['sentences'][i]
            text = nltk.word_tokenize(sentence.strip().lower())
            parsed_sentence = nltk.pos_tag(text)
            out = gen(text, parsed_sentence, isTrain)
            if len(out)==2:
                masked_sentence, label = out
                data.append([key, start_frame, end_frame, masked_sentence, label])
    
    sorted_words = sorted(wordDict.items(), key= lambda k : k[1]["freq"], reverse=True)

    for i in range(len(data)):
        featureDim = 1000
        textFeature = torch.zeros(1, featureDim)
        for j in range(featureDim):
            for word in data[i][3]:
                if word == sorted_words[j][0]:
                    textFeature[0, j] += 1
            # if sorted_words[j][0] in data[i][3]:
            #     textFeature[0,j] = 1
        data[i].append(textFeature)
    
    return data

def getFeatures(textData):
    features = []
    for data in textData:
        videoFeature = getVideoFeatures(data[0], data[1], data[2])
        if data[4] in wordDict:
            label = torch.tensor([wordDict[data[4]]["id"]])
        else:
            label = torch.tensor([0])
        features.append([data[5], videoFeature, label])
    return features


def train(data, max_epoch, model, optimizer, criterion):
    
    model.train()
    running_loss = 0
    for epoch in range(max_epoch):
        for n, batch in enumerate(data):
            optimizer.zero_grad()
            text_feature, video_feature, label = batch           
            
            output = model(text_feature, video_feature)
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
        output = model(text_feature, video_feature)
        if torch.argmax(output).item == label.item():
            correct += 1
        total_num += 1

    acc = correct / total_num
    print(acc)

trainText = getTextFeatures(trainTextFile)
trainFeatures = getFeatures(trainText)
print("successfully load train")

valText = getTextFeatures(valTextFile, isTrain=False)
valFeatures = getFeatures(valText)
print("successfully load val")

model = baseline_BOW_VF(1000, 500, len(wordDict))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
max_epoch = 3

model = train(trainFeatures, max_epoch, model, optimizer, criterion)
evaluation(valFeatures, model)


