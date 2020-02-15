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

# answerWordDict = {}
# THRESHOLD = 500

# def getVideoFeatures(key, startFrame, endFrame):
#     feature_h5 = videoFeature[key]['c3d_features']
#     shape = feature_h5.shape
#     feature_np = np.zeros(shape)
#     feature_h5.read_direct(feature_np)
#     feature = feature_np.mean(axis=0)
#     return torch.FloatTensor(feature).view(1, -1)

# def gen(text, parsed_sentence, isTrain):
#     position = []
#     for i in range(len(parsed_sentence)):
#         if parsed_sentence[i][1] == 'JJ' or parsed_sentence[i][1] == 'NN':
#             position.append(i)

#     while len(position): 
#         idx = random.sample(position, 1)[0]
#         if text[idx] in answerWordDict and answerWordDict[text[idx]] >= THRESHOLD and isTrain:
#             position.remove(idx)
#             continue
#         new_sentence = text[:]

#         correct_word = new_sentence[idx]
#         if isTrain:
#             if correct_word in answerWordDict:
#                 answerWordDict[correct_word] += 1
#             else:
#                 answerWordDict[correct_word] = 1

#         new_sentence[idx] = '[MASK]'
#         return new_sentence, correct_word
    
#     return ()

# def getTextFeatures(textFile, isTrain=True):
#     with open(textFile, 'r') as f:
#         raw = json.load(f)
    
#     data = []

#     for key in raw.keys():
#         total_events = len(raw[key]['sentences'])
#         for i in range(total_events):
#             start_frame = math.floor(raw[key]['timestamps'][i][0] * 2)
#             end_frame = math.ceil(raw[key]['timestamps'][i][1] * 2)
#             sentence = raw[key]['sentences'][i]
#             text = nltk.word_tokenize(sentence.strip().lower())
#             parsed_sentence = nltk.pos_tag(text)
#             out = gen(text, parsed_sentence, isTrain)
#             if len(out)==2:
#                 masked_sentence, label = out
#                 data.append([key, start_frame, end_frame, masked_sentence, label])
    
#     sorted_words = sorted(wordDict.items(), key= lambda k : k[1]["freq"], reverse=True)

#     for i in range(len(data)):
#         featureDim = 1000
#         textFeature = torch.zeros(1, featureDim)
#         for j in range(featureDim):
#             for word in data[i][3]:
#                 if word == sorted_words[j][0]:
#                     textFeature[0, j] += 1
#             # if sorted_words[j][0] in data[i][3]:
#             #     textFeature[0,j] = 1
#         data[i].append(textFeature)
    
#     return data

# def getFeatures(textData):
#     features = []
#     for data in textData:
#         videoFeature = getVideoFeatures(data[0], data[1], data[2])
#         if data[4] in wordDict:
#             label = torch.tensor([wordDict[data[4]]["id"]])
#         else:
#             print(data[4])
#             label = torch.tensor([21086]) #should not happen just in case
#         features.append([data[5], videoFeature, label])
#     return features


def train(data, max_epoch, model, optimizer, criterion):
    
    model.train()
    running_loss = 0
    for epoch in range(max_epoch):
        for n, batch in enumerate(data):
            optimizer.zero_grad()
            text_feature, video_feature, label = batch           
            
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
trainLoader = DataLoader(trainDataset, batch_size=16, shuffle=True, num_workers=4)
print("successfully load train")

valDataset = ActivityNetCaptionDataset(valTextFile, videoFeatures, word_dict, isTrain=False)
valLoader = DataLoader(valDataset, batch_size=16, shuffle=True, num_workers=4)
print("successfully load val")

model = baseline_BOW_VF(1000, 500, len(word_dict))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
max_epoch = 3

model = train(trainLoader, max_epoch, model, optimizer, criterion)
evaluation(valLoader, model)


