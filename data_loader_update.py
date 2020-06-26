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

    # data list will be saved into pickle file - each element is an tuple:
    # (sentence_representation, feature representation, label_representation)
    data = []
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
                mask_position = sentence.split(' ').index('[MASK]') + 1
                label = train_file.readline().strip()
                tt_start, tt_end = json.loads(train_file.readline().strip())
                tt_start, tt_end = float(tt_start), float(tt_end)
                duration = raw_data[key]["duration"]

                video_feature_len = video_features[key]['c3d_features'].shape[0]
                start_frame = math.floor(tt_start / duration * video_feature_len)
                end_frame  = math.floor(tt_end / duration * video_feature_len)

                video_feature_np = video_features[key]['c3d_features'][startFrame:endFrame+1]
        
                if vidoe_feature_np.shape[0] > 200:
                    video_feature = np.zeros((200, video_feature_np.shape[1]))
                    for i in range(200):
                        video_feature[i] = video_feature_np[round(i * (video_feature_np.shape[0]-1)/199)]
                else:
                    video_feature = video_feature_np

                video_feature = torch.tensor(video_feature, dtype=torch.float)
                data.append([sentence, video_feature, label, mask_position, key])
    return data

folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"

video_features = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
genTextFile = f"{folder}/train"
rawTextFile = f"{folder}/train.json"

data = gen(genTextFile, rawTextFile, video_features)

with open('train.pkl', 'wb') as f:
    pickle.dump(data, f)