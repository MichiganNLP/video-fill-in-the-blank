import pickle
import os
import h5py
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

from baseline_BOW_VF import baseline_BOW_VF
from data_loader_multimodal import ActivityNetCaptionsDataset

print("Very begin")
folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"
videoFeatures = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
trainTextFile = f"{folder}/train.json"
valTextFile = f"{folder}/val_1.json"


stop_words = set(stopwords.words('english'))

# small samples testing
# trainTextFile = 'train.json'
# valTextFile = 'val.json'


def fit(train_text_file, num_tokens):
    word_to_idx = dict()
    file = open(train_text_file, 'r')
    data = json.load(file)
    # collect all documents into all_sentences
    documents = list()
    for key in data:
        documents.extend(data[key]['sentences'])
    # tokenize documents
    tokenzied_docs = [word_tokenize(doc.lower()) for doc in documents]
    # filter out stop words
    tokens = [w for t_doc in tokenzied_docs for w in t_doc if w.isalpha()]
    tokens_no_stops = [t for t in tokens if t not in stop_words]
    # Get top k tokens by frequency
    top_k_words = Counter(tokens_no_stops).most_common(num_tokens)
    # create (word, index) dictionary
    for word, _ in top_k_words:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
    return word_to_idx


def transform(sentence, word_to_idx):
    vector = torch.zeros(len(word_to_idx))
    tokens = [w for w in word_tokenize(sentence) if w.isalpha()]
    token_no_stops = [t for t in tokens if t not in stop_words]
    for word in token_no_stops:
        if word in word_to_idx:
            vector[word_to_idx[word]] += 1
    return vector.view(1, -1)


def train(data, max_epoch, model, optimizer, criterion):
    model.train()
    running_loss = 0
    for epoch in range(max_epoch):
        for n, batch in enumerate(data):
            optimizer.zero_grad()
            text_feature, video_feature, label = batch

            # use GPU
            text_feature = text_feature.cuda()
            video_feature = video_feature.cuda()
            label = label.cuda()

            output = model(text_feature, video_feature)
            # output = output.squeeze(dim=1)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if n % 50 == 0 and n != 0:
                print("Epoch {}, batch {}: loss = {}".format(epoch, n, running_loss / 50))
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

trainDataset = ActivityNetCaptionsDataset(trainTextFile, videoFeatures)
trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True, num_workers=4)
print("successfully load train")

valDataset = ActivityNetCaptionsDataset(valTextFile, videoFeatures, isTrain=False)
valLoader = DataLoader(valDataset, batch_size=16, shuffle=True, num_workers=4)
print("successfully load val")

model = baseline_BOW_VF(1000, 500, len(word_dict)).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
max_epoch = 30

model = train(trainLoader, max_epoch, model, optimizer, criterion)
evaluation(valLoader, model)
