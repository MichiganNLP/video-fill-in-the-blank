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
import csv

import csv
import pickle

class ActivityNetCaptionDataset(Dataset):

    def __init__(self, textFile, videoFeatures, duration_file, isTrain=True):
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
        self.one_token = 0
        self.two_tokens = 0
        self.threemore_tokens = 0

        textFeature, self.out_text = self.getTextFeatures(textFile, isTrain)
        
        with open("duration_file", 'rb') as f:
            self.duration = pickle.load(f)

        self.data = self.getFeatures(textFeature, videoFeatures)

        # with open('train', 'w') as f:
        #     for line in self.out_text:
        #         f.write(line[0])
        #         f.write('\n')
        #         f.write(' '.join(line[1]))
        #         f.write('\n')
        #         f.write(line[2])
        #         f.write('\n')
        #         f.write(line[3])
        #         f.write('\n')
        #         f.write(str(line[4]))
        #         f.write('\n')
        #         f.write('\n')
        
        # with open('train.csv', 'w') as csvfile:
        #     fieldnames = ['question', 'video_id', 'pos_tag', 'video_start_time', 'video_end_time', 'answer']
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')

        #     writer.writeheader()
        #     for line in self.out_text:
        #         writer.writerow({'question': ' '.join(line[1]), 'video_id': line[0], 'pos_tag': line[3],
        #          'video_start_time':str(line[4][0]), 'video_end_time': str(line[4][1]), 'answer':line[2]})

        # print(self.one_token, self.two_tokens, self.threemore_tokens, len(self.data))
        with open('train.pkl', 'wb') as f:
            pickle.dump(self.data, f)


    def getVideoFeatures(self, key, startTime, endTime, videoFeatures, textLen):
        video_feature_len = videoFeatures[key]['c3d_features'].shape[0]

        duration = float(self.duration[key])

        startFrame = math.floor(startTime / duration * video_feature_len)
        endFrame  = math.floor(endTime / duration * video_feature_len)
        
        feature_np = videoFeatures[key]['c3d_features'][startFrame:endFrame+1]
        # shape = feature_h5.shape
        # feature_np = np.zeros(shape)
        # feature_h5.read_direct(feature_np)
        
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
                position.append((i, parsed_sentence[i][1]))

        while len(position): 
            idx, POS = random.sample(position, 1)[0]
            if text[idx] in self.answerWordDict and self.answerWordDict[text[idx]] >= self.THRESHOLD and isTrain:
                position.remove((idx, POS))
                continue
            new_sentence = text[:]
            correct_word = new_sentence[idx]
            correct_word_tokenized = self.tokenizer.tokenize(correct_word)
            if isTrain:
                if correct_word in self.answerWordDict:
                    self.answerWordDict[correct_word] += 1
                else:
                    self.answerWordDict[correct_word] = 1

            correct_word_len = len(correct_word_tokenized)

            if correct_word_len == 1:
                self.one_token += 1
            elif correct_word_len == 2:
                self.two_tokens += 1
            else:
                self.threemore_tokens += 1

            if isTrain:
                sentence_for_model = new_sentence[0:idx] + ['[MASK]'] * correct_word_len + new_sentence[idx+1:]
            else:
                sentence_for_model = new_sentence[0:idx] + ['[MASK]'] + new_sentence[idx + 1:]
            new_sentence[idx] = '[MASK]'

            sequence_id = self.tokenizer.encode(' '.join(sentence_for_model))
            return sequence_id, correct_word_tokenized, idx+1, new_sentence, correct_word, POS
        return ()

    # This commented function was used to extract text features from original ActivityNet Caption data
    # def getTextFeatures(self, textFile, isTrain=True):
    #     with open(textFile, 'r') as f:
    #         raw = json.load(f)
        
    #     data = []
    #     out_text = []
    #     # debug_count = 0
    #     for key in raw.keys():
    #         # if debug_count >= 500 and isTrain:
    #         #     break
    #         # if debug_count >= 100 and not isTrain:
    #         #     break
    #         # debug_count += 1
    #         total_events = len(raw[key]['sentences'])
    #         for i in range(total_events):
    #             start_frame = math.floor(raw[key]['timestamps'][i][0] * 2)
    #             end_frame = math.ceil(raw[key]['timestamps'][i][1] * 2)
    #             sentence = raw[key]['sentences'][i]
    #             text = nltk.word_tokenize(sentence.strip().lower())
    #             parsed_sentence = nltk.pos_tag(text)
    #             out = self.gen(text, parsed_sentence, isTrain)
    #             if len(out)==6:
    #                 masked_sentence, label, masked_position, original_sentence, correct_word, POS = out
    #                 data.append([key, start_frame, end_frame, masked_sentence, label, masked_position])
    #                 out_text.append([key, original_sentence, correct_word, POS, raw[key]['timestamps'][i]])
                
    #     return data, out_text

    def getTextFeatures(self, textFile, isTrain=True):
        data = []
        out_text = []
        
        with open(textFile, 'r') as csv_file:
            csv_reader = csv.reader(csv_file,delimiter='\t')
            isHead = True
            for row in csv_reader:
                if (isHead):
                    isHead = False
                key = row[0]
                start_time = row[4]
                end_time = row[5]
                sentence = row[1]
                text = nltk.word_tokenize(sentence.strip().lower())
                parsed_sentence = nltk.pos_tag(text)
                out = self.gen(text, parsed_sentence, isTrain)
                if len(out)==6:
                    masked_sentence, label, masked_position, original_sentence, correct_word, POS = out
                    data.append([key, start_time, end_time, masked_sentence, label, masked_position])
                    out_text.append([key, original_sentence, correct_word, POS, raw[key]['timestamps'][i]])
                
        return data, out_text

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
