# This file defines a data loader that generate pickle data from the text file generated for MTurk
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
import pickle

def gen(masked_data_file, text_file, video_features):
    # masked_data_file: Generated questions with [MASK]
    # text_file: Original data file
    # video_features: hd5 video features

    data = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open(text_file, 'r') as f:
        raw_data = json.load(f)

    with open(masked_data_file, 'r') as train_file:
        with open(text_file, 'r') as raw_file:
            while True:
                key = train_file.readline().strip()
                # check if it is the end of train file
                if not key:
                    break
                sentence = train_file.readline().strip()
                sentence_id = tokenizer.encode(sentence) 
                mask_position = sentence.split(' ').index('[MASK]') + 1

                label = tokenizer.tokenize(train_file.readline().strip())

                tt_start, tt_end = json.loads(train_file.readline().strip())
                tt_start, tt_end = float(tt_start), float(tt_end)
                duration = raw_data[key]["duration"]

                train_file.readline()

                video_feature_len = video_features[key]['c3d_features'].shape[0]
                start_frame = math.floor(tt_start / duration * video_feature_len)
                end_frame  = math.floor(tt_end / duration * video_feature_len)

                video_feature_np = video_features[key]['c3d_features'][start_frame:end_frame+1]
        
                if video_feature_np.shape[0] > 200:
                    video_feature = np.zeros((200, video_feature_np.shape[1]))
                    for i in range(200):
                        video_feature[i] = video_feature_np[round(i * (video_feature_np.shape[0]-1)/199)]
                else:
                    video_feature = video_feature_np

                video_feature = torch.tensor(video_feature, dtype=torch.float)
                data.append([sentence_id, video_feature, label, mask_position, key])
    return data

folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"

video_features = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')

# Train
genTextFile_train = f"{folder}/val2"
rawTextFile_train = f"{folder}/val_2.json"

data = gen(genTextFile_train, rawTextFile_train, video_features)

with open(f'{folder}/train.pkl', 'wb') as f:
    pickle.dump(data, f)

# Val
genTextFile_val1 = f"{folder}/val1"
rawTextFile_val1 = f"{folder}/val_1.json"

data = gen(genTextFile_val1, rawTextFile_val1, video_features)

with open(f'{folder}/val1.pkl', 'wb') as f:
    pickle.dump(data, f)

# Test
genTextFile_val2 = f"{folder}/val2"
rawTextFile_val2 = f"{folder}/val_2.json"

data = gen(genTextFile_val2, rawTextFile_val2, video_features)

with open(f'{folder}/val2.pkl', 'wb') as f:
    pickle.dump(data, f)