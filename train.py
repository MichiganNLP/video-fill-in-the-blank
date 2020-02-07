import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import ActivityNetCaptionDataset

import numpy as np
import h5py
import json
import math
import nltk
from baseline_BOW_VF import baseline_BOW_VF

videoFeature = h5py.File("/scratch/mihalcea_root/mihalcea1/ruoyaow/ActivityNet_Captions/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
# trainTextFile = "/scratch/mihalcea_root/mihalcea1/ruoyaow/ActivityNet_Captions/train.json"
# valTextFile = "/scratch/mihalcea_root/mihalcea1/ruoyaow/ActivityNet_Captions/val_1.json"
trainTextFile = 'train.json'
valTextFile = 'val.json'

wordDict = {}
answerWordDict = {}
THRESHOLD = 500
wordID = 0

def getVideoFeatures(key, startFrame, endFrame):
    feature_h5 = videoFeature[key]['c3d_features']
    shape = feature_h5.shape
    feature_np = np.zeros(shape)
    feature_h5.read_direct(feature_np)
    feature = feature_np.mean(axis=0)
    return torch.tensor(feature)

def gen(text, parsed_sentence):
    position = []
    for i in range(len(parsed_sentence)):
        if parsed_sentence[i][1] == 'JJ' or parsed_sentence[i][1] == 'NN':
            position.append(i)

    while len(position): 
        idx = random.sample(position, 1)[0]
        if text[idx] in answerWordDict and answerWordDict[text[idx]] >= THRESHOLD:
            position.remove(idx)
            continue
        new_sentence = text[:]

        correct_word = new_sentence[idx]
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

def getTextFeatures(textFile):
    with open(textFile, 'r') as f:
        raw = json.load(f)
    
    data = []
    for key in raw.keys():
        total_events = len(raw[key]['sentences'])
        for i in range(total_events):
            start_frame = math.floor(raw[key]['timestamps'][i][0] * 2)
            end_frame = math.ceil(raw[key]['timestamps'][i][1] * 2)
            sentence = raw[key]['sentences'][i]
            text = nltk.word_tokenize(sentence.strip().lower())
            parsed_sentence = nltk.pos_tag(text)
            masked_sentence, label = gen(text, parsed_sentence)
            data.append([key, start_frame, end_frame, masked_sentence, label])
    
    sorted_words = sorted(wordDict.items(), key= lambda k : k[1]["freq"], reverse=True)

    for i in range(len(data)):
        textFeature = torch.zeros(1, 1000)
        for j in range(1000):
            if sorted_words[j][0] in data[i][3]:
                textFeature[j] = 1
        data[i].append(textFeature)
    
    return data

def getFeatures(textData):
    features = []
    for data in textData:
        videoFeature = getVideoFeatures(data[0], data[1], data[2])
        features.append(data[5], videoFeature, torch.tensor(wordDict[data[4]]["id"]))
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

valText = getTextFeatures(valTextFile)
valFeatures = getFeatures(valText)

model = baseline_BOW_VF(1000, 500, len(wordDict))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
max_epoch = 10

model = train(trainFeatures, max_epoch, model, optimizer, criterion)
evaluation(valFeatures, model)


