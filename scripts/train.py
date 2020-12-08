import pickle

import torch
import torch.nn as nn
from baseline_BOW_VF import baseline_BOW_VF
from torch.utils.data import DataLoader

from lqam.data_loader_multimodal import ActivityNetCaptionsDataset
from lqam.iterable_utils import build_representation, fit


def train(data, max_epoch, model, optimizer, criterion):
    model.train()
    running_loss = 0
    for epoch in range(max_epoch):
        for n, batch in enumerate(data):

            optimizer.zero_grad()

            text_feature, video_feature, labels = batch
            # use GPU
            text_feature = text_feature.cuda()
            video_feature = video_feature.cuda()
            labels = labels.squeeze(dim=1)
            labels = labels.cuda()

            output = model(text_feature, video_feature)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if n % 50 == 49:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, n + 1, running_loss / 50))
                running_loss = 0
    return model


def evaluation(test_data, model):
    model.eval()
    correct = 0
    total_num = 0

    for data in test_data:
        text_feature, video_feature, labels = data
        text_feature = text_feature.cuda()
        video_feature = video_feature.cuda()
        _, labels = labels.max(dim=1)
        labels = labels.cuda()

        output = model(text_feature, video_feature)
        output = output.squeeze(dim=1)
        batch_size = output.shape[0]
        for i in range(batch_size):
            if torch.argmax(output[i]).item() == labels[i].item():
                correct += 1
            total_num += 1

    acc = correct / total_num
    print(f'accuracy: {acc}')


folder = '/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions'
video_features_file = f'{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5'
train_text_file = f'{folder}/train.json'

bow_representation_train_file = f'{folder}/bow_training_data.pkl'
bow_representation_val_file = f'{folder}/bow_validation_data.pkl'
masked_training_data_file = f'{folder}/train'
masked_validation_data_file = f'{folder}/val1'

num_tokens = 1000

print('start constructing bag of word representations for training and validation using current setting')
print(f'num_tokens = {num_tokens}, which represents the number of most frequent words used in text representations')
print(f'a list of tuples (sentence representation, video feature, label) '
      f'\n-- training data will be saved in {bow_representation_train_file},'
      f'\n-- validation data will be saved in {bow_representation_val_file}')

build_representation(num_tokens, masked_training_data_file, train_text_file,
                     video_features_file, bow_representation_train_file)
build_representation(num_tokens, masked_validation_data_file, train_text_file,
                     video_features_file, bow_representation_val_file)

print('loading training data...')
trainDataset = ActivityNetCaptionsDataset(bow_representation_train_file)
trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True, num_workers=0)
print("successfully loaded training data")

valDataset = ActivityNetCaptionsDataset(bow_representation_val_file)
valLoader = DataLoader(valDataset, batch_size=16, shuffle=True, num_workers=0)
print("successfully loaded validation data")

_, num_vocabs = fit(train_text_file, 1000)
print(f"num_vocabs: {num_vocabs}")
model = baseline_BOW_VF(1000, 500, num_vocabs).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00003)
max_epoch = 30

model = train(trainLoader, max_epoch, model, optimizer, criterion)
evaluation(valLoader, model)

model_file = f'{folder}/baseline_bow_average_video_feature_model.pkl'
model_pickle_file = open(model_file, 'wb')
pickle.dump(model, model_pickle_file)
model_pickle_file.close()
print(f"successfully saved the model in {model_file}")
