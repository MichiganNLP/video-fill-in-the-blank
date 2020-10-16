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

PATH = 'Checkpoint'
folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"
verb_blacklist = ["can", "could", "is", "am", "are", "was", "were", "be", "been", "has", "had", "have", "seen"]
verb_pos = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
noun_pos = ["NN", "NNS"]

videoFeatures = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
trainTextFile = "Train.tsv"
valTextFile = "Validation.tsv"
testTextFile = "Test.tsv"
testMTurkFile = "Test_mturk.tsv"
valMTurkFile = "Validation_mturk.tsv"
# trainTextFile = "TrainSubset.csv"
# valTextFile = "ValidationSubset.csv"
# testTextFile = "TestSubset.csv"
duration_file = f"{folder}/latest_data/multimodal_model/video_duration.pkl"

name = "train"
if name == "train":
    textFile = trainTextFile
    mturkTextFile = None
    isTrain = True
elif name == "val":
    textFile = valTextFile
    mturkTextFile = valMTurkFile
    isTrain = False
elif name == "test":
    textFile = testTextFile
    mturkTextFile = testMTurkFile
    isTrain = False
answerWordDict = {}
THRESHOLD = 500
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

with open(duration_file, 'rb') as f:
    duration = pickle.load(f)

def getMturkQuestions(mturkTextFile):
    if isTrain:
        return None
    mturkQuestions = []
    with open(mturkTextFile, 'r') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter='\t')
        isHead = True
        for row in csv_reader:
            if isHead:
                isHead = False
                continue
            mturkQuestions.append(row[1])
    return mturkQuestions

def getVideoFeatures(key, startTime, endTime, videoFeatures, textLen):
    video_feature_len = videoFeatures[key]['c3d_features'].shape[0]

    duration_time = float(duration[key])

    startFrame = math.floor(startTime / duration_time * video_feature_len)
    endFrame  = math.floor(endTime / duration_time * video_feature_len)
    
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

def gen(text, isTrain):
    position = []    
    parsed_sentence = nltk.pos_tag(text)
    for i in range(len(parsed_sentence)):
        if (parsed_sentence[i][1] in verb_pos and parsed_sentence[i][0] not in verb_blacklist) or parsed_sentence[i][1] == 'JJ' or parsed_sentence[i][1] in noun_pos:
            position.append((i, parsed_sentence[i][1]))

    while len(position): 
        idx, POS = random.sample(position, 1)[0]
        if text[idx] in answerWordDict and answerWordDict[text[idx]] >= THRESHOLD and isTrain:
            position.remove((idx, POS))
            continue
        new_sentence = text[:]
        correct_word = new_sentence[idx]
        correct_word_tokenized = tokenizer.tokenize(correct_word)
        if isTrain:
            if correct_word in answerWordDict:
                answerWordDict[correct_word] += 1
            else:
                answerWordDict[correct_word] = 1

        correct_word_len = len(correct_word_tokenized)

        if isTrain:
            sentence_for_model = new_sentence[0:idx] + ['[MASK]'] * correct_word_len + new_sentence[idx+1:]
        else:
            sentence_for_model = new_sentence[0:idx] + ['[MASK]'] + new_sentence[idx + 1:]
        new_sentence[idx] = '[MASK]'

        sequence_id = tokenizer.encode(' '.join(sentence_for_model))
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

def getTextFeatures(textFile, mturkQuestionList, isTrain=True):
    data = []
    out_text = []
    
    with open(textFile, 'r') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter='\t')
        isHead = True
        for row in csv_reader:
            if (isHead):
                isHead = False
                continue
            key = row[0] # v_*
            start_time = float(row[4])
            end_time = float(row[5])
            if not isTrain and row[1] in mturkQuestionList:
                text = tokenizer.tokenize(row[1])
                masked_position = text.index('[MASK]') + 1 # +1 for [CLS]
                label = tokenizer.tokenize(row[2])
                correct_word_len = len(label)
                if isTrain:
                    for _ in range(correct_word_len - 1):
                        text.insert(masked_position, '[MASK]')

                sequence_id = tokenizer.encode(' '.join(text))        

                data.append([key, start_time, end_time, sequence_id, label, masked_position])
                out_text.append([key, row[1], row[2], row[3],[start_time, end_time]])
            else:                
                text = row[1].strip().split()
                masked_position = text.index('[MASK]')
                label = row[2]
                text[masked_position] = label
                text = nltk.word_tokenize(' '.join(text))

                out = gen(text, isTrain)
                if len(out) != 0:
                    sequence_id, label, masked_position, question_text, label_text, POS = out        

                    data.append([key, start_time, end_time, sequence_id, label, masked_position])
                    out_text.append([key, question_text, label_text, POS, [start_time, end_time]])
            
    return data, out_text

def getFeatures(textData, videoFeatures):
    features = []
    for data in textData:
        textLen = len(data[3])
        videoFeature = getVideoFeatures(data[0], data[1], data[2], videoFeatures, textLen)
        features.append([data[3], videoFeature, data[4], data[5],data[0]])
    return features

mturk_question_list = getMturkQuestions(mturkTextFile)
textFeature, out_text = getTextFeatures(textFile, mturk_question_list, isTrain)



data = getFeatures(textFeature, videoFeatures)

with open(f'{name}_verb.csv', 'w') as csvfile:
    fieldnames = ['question', 'video_id', 'pos_tag', 'video_start_time', 'video_end_time', 'answer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')

    writer.writeheader()
    for line in out_text:
        writer.writerow({'question': ' '.join(line[1]), 'video_id': line[0], 'pos_tag': line[3],
         'video_start_time':str(line[4][0]), 'video_end_time': str(line[4][1]), 'answer':line[2]})

with open(f'/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/latest_data/multimodal_model/verb_data/{name}.pkl', 'wb') as f:
    pickle.dump(data, f)