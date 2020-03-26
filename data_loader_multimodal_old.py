from torch.utils.data import Dataset
import torch
from nltk.corpus import wordnet
import json
import math
import nltk
import random
import numpy as np
import h5py
from transformers import BertTokenizer, BertModel

import pickle

class ActivityNetCaptionDataset(Dataset):

    def __init__(self, textFile, videoFeatures, isTrain=True):
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
        self.answerWordDict = {}
        self.isTrain = isTrain
        self.THRESHOLD = 500
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        textFeature = self.getTextFeatures(textFile, isTrain)
        self.data = self.getFeatures(textFeature, videoFeatures)

        with open('val.pkl', 'wb') as f:
            pickle.dump(self.data, f)

    def getVideoFeatures(self, key, startFrame, endFrame, videoFeatures, textLen):
        feature_h5 = videoFeatures[key]['c3d_features']
        shape = feature_h5.shape
        feature_np = np.zeros(shape)
        feature_h5.read_direct(feature_np)
        
        if feature_np.shape[0] > 200:
            feature = np.zeros((200, feature_np.shape[1]))
            for i in range(200):
                feature[i] = feature_np[round(i * (feature_np.shape[0]-1)/199)]
        else:
            feature = feature_np

        return torch.tensor(feature, dtype=torch.float)
    
    def gen(self, text, parsed_sentence, isTrain):
        position = []
        for i in range(len(parsed_sentence)):
            if parsed_sentence[i][1] == 'JJ' or parsed_sentence[i][1] == 'NN':
                position.append(i)

        while len(position): 
            idx = random.sample(position, 1)[0]
            if text[idx] in self.answerWordDict and self.answerWordDict[text[idx]] >= self.THRESHOLD and isTrain:
                position.remove(idx)
                continue
            new_sentence = text[:]
            correct_word = new_sentence[idx]
            if isTrain:
                if correct_word in self.answerWordDict:
                    self.answerWordDict[correct_word] += 1
                else:
                    self.answerWordDict[correct_word] = 1

            new_sentence[idx] = '[MASK]'
            sequence_id = self.tokenizer.encode(' '.join(new_sentence))
            return sequence_id, correct_word, idx+1
        
        return ()

    def getTextFeatures(self, textFile, isTrain=True):
        with open(textFile, 'r') as f:
            raw = json.load(f)
        
        data = []
        # debug_count = 0
        for key in raw.keys():
            # if debug_count >= 500 and isTrain:
            #     break
            # if debug_count >= 100 and not isTrain:
            #     break
            # debug_count += 1
            total_events = len(raw[key]['sentences'])
            for i in range(total_events):
                start_frame = math.floor(raw[key]['timestamps'][i][0] * 2)
                end_frame = math.ceil(raw[key]['timestamps'][i][1] * 2)
                sentence = raw[key]['sentences'][i]
                text = nltk.word_tokenize(sentence.strip().lower())
                parsed_sentence = nltk.pos_tag(text)
                out = self.gen(text, parsed_sentence, isTrain)
                if len(out)==3:
                    masked_sentence, label, masked_position = out
                    data.append([key, start_frame, end_frame, masked_sentence, label, masked_position])
                
        return data

    def getFeatures(self, textData, videoFeatures):
        features = []
        for data in textData:
            textLen = len(data[3])
            videoFeature = self.getVideoFeatures(data[0], data[1], data[2], videoFeatures, textLen)
            features.append([data[3], videoFeature, data[4], data[5],data[0]])
        return features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 
