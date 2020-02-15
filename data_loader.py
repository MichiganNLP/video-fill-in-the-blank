from torch.utils.data import Dataset
import torch
from nltk.corpus import wordnet
import json
import math
import nltk
import random
import numpy as np
import h5py


class ActivityNetCaptionDataset(Dataset):

    def __init__(self, textFile, videoFeatures, word_dict, isTrain=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.word_dict = word_dict
        self.answerWordDict = {}
        self.isTrain = isTrain
        self.THRESHOLD = 500

        textFeature = self.getTextFeatures(textFile, isTrain)
        self.data = self.getFeatures(textFeature, videoFeatures)

    def getVideoFeatures(self, key, startFrame, endFrame, videoFeatures):
        feature_h5 = videoFeatures[key]['c3d_features']
        shape = feature_h5.shape
        feature_np = np.zeros(shape)
        feature_h5.read_direct(feature_np)
        feature = feature_np.mean(axis=0)
        return torch.FloatTensor(feature).view(-1)
    
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
            return new_sentence, correct_word
        
        return ()

    def getTextFeatures(self, textFile, isTrain=True):
        with open(textFile, 'r') as f:
            raw = json.load(f)
        
        data = []
        debug_count = 0
        for key in raw.keys():
            if debug_count >= 1000 and isTrain:
                break
            if debug_count >= 100 and not isTrain:
                break
            debug_count += 1
            total_events = len(raw[key]['sentences'])
            for i in range(total_events):
                start_frame = math.floor(raw[key]['timestamps'][i][0] * 2)
                end_frame = math.ceil(raw[key]['timestamps'][i][1] * 2)
                sentence = raw[key]['sentences'][i]
                text = nltk.word_tokenize(sentence.strip().lower())
                parsed_sentence = nltk.pos_tag(text)
                out = self.gen(text, parsed_sentence, isTrain)
                if len(out)==2:
                    masked_sentence, label = out
                    data.append([key, start_frame, end_frame, masked_sentence, label])
        
        sorted_words = sorted(self.word_dict.items(), key= lambda k : k[1]["freq"], reverse=True)

        for i in range(len(data)):
            featureDim = 1000
            textFeature = torch.zeros(featureDim)
            for j in range(featureDim):
                for word in data[i][3]:
                    if word == sorted_words[j][0]:
                        textFeature[j] += 1
                # if sorted_words[j][0] in data[i][3]:
                #     textFeature[0,j] = 1
            data[i].append(textFeature)
        
        return data

    def getFeatures(self, textData, videoFeatures):
        features = []
        for data in textData:
            videoFeature = self.getVideoFeatures(data[0], data[1], data[2], videoFeatures)
            if data[4] in self.word_dict:
                label = torch.tensor(self.word_dict[data[4]]["id"])
            else:
                print(data[4])
                label = torch.tensor(21086) #should not happen just in case
            features.append([data[5], videoFeature, label])
        return features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 