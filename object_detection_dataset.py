from torch.utils.data import Dataset
import torch
import math
import numpy as np
from transformers import BertTokenizer, BertModel

import csv
import pickle

class ObjectDetectionDataset(Dataset):

    def __init__(self, textFile, videoFolder, name, isTrain=True):
        """
        Args:
            textFile: text file path
            videoFeatures: video feature hd5 data
            isTrain: true for training, false for eval
        Output data structure:
            masked sentence
            video feature
            label
            mask position
            video url
        """
        self.isTrain = isTrain
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.videoFolder = videoFolder

        self.textFeatures = self.getTextFeatures(textFile, isTrain)

    def getTextFeatures(self, textFile, isTrain=True):
        data = []        
        with open(textFile, 'r') as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=',')
            isHead = True
            for row in csv_reader:
                if (isHead):
                    isHead = False
                    continue
                key = row[0] # v_*
                start_time = float(row[4])
                end_time = float(row[5])
                text = self.tokenizer.tokenize(row[1])
                masked_position = text.index('[MASK]') + 1 # +1 for [CLS]
                label = self.tokenizer.tokenize(row[2])
                correct_word_len = len(label)
                if isTrain:
                    for _ in range(correct_word_len - 1):
                        text.insert(masked_position, '[MASK]')

                sequence_id = self.tokenizer.encode(' '.join(text))
                data.append([key, start_time, end_time, sequence_id, label, masked_position])
        return data

    def aggregate(self, x, key):
        x = x.view(x.shape[0], -1)
        x = torch.mean(x, dim=0)
        x = x.view(1, -1)
        return x

    def __len__(self):
        return len(self.textFeatures)

    def __getitem__(self, idx):
        key, start_time, end_time, text, label, mask_position = self.textFeatures[idx]
        with open(f"{self.videoFolder}/{key}.pkl", 'rb') as f:
            videoFeatures = pickle.load(f)

        start_idx = math.floor(start_time * 5)
        end_idx = math.floor(end_time * 5)
        videoFeatures = videoFeatures[start_idx : end_idx + 1]
        if len(videoFeatures) > 200:
            feature = []
            for i in range(200):
                feature.append(videoFeatures[round(i * (len(videoFeatures)-1)/199)])
            videoFeatures = feature
        
        debug = 0
        box_list = [self.aggregate(videoFeature[0][0], key) for videoFeature in videoFeatures if videoFeature[0][0].shape[0] != 0]
        box_feature_list = [self.aggregate(videoFeature[1][0], key) for videoFeature in videoFeatures if videoFeature[0][0].shape[0] != 0]
        if len(box_list) != 0:
            boxes = torch.cat(box_list, 0)
            box_features = torch.cat(box_feature_list, 0)
        else:
            boxes = None
            box_features = None

        return text, box_features, boxes, label, mask_position
        



