# This file preprocess data from Mturk and dumps them into a pickle file for further usage
# Output data structure:
#   (masked_sentence, video_features, extended_answers, mask_position, standard_answer)
#       masked_sentence: sequence id of the masked sentence. [CLS] and [SEP] already added
#       video_features: C3D video features. T * 500
#       extended answers: set of valid MTurk workers' answers
#       mask_position: index of the [MASK] token in the question
#       standard_answer: original answer

import csv
import h5py
import math
import torch
import numpy as np
import pickle
import json
from transformers import BertTokenizer, BertModel

import os

folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
video_features = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')

file_names = os.listdir(f'{folder}/MTurk_data/val1/')

for csvData in file_names: 
    data = []

    text_file = f"{folder}/val_1.json"
    with open(text_file, 'r') as f:
        raw_data = json.load(f)


    with open(f"{folder}/MTurk_data/val1/{csvData}") as csvfile:
        reader = csv.reader(csvfile)
        isHead = True
        for row in reader:
            if isHead:
                isHead = False
                continue
            video_id, question, start_time, end_time, _, standard_answer, worker_answers = row
            
            # original worker answers is a dict-like string, convert it to a real dictionary        
            worker_answers = eval(worker_answers)
            if worker_answers != []:
                extended_answers = list(set([standard_answer] + list(worker_answers.keys())))
            else:
                extended_answers = [standard_answer]
            masked_sentence = tokenizer.tokenize(question)
            mask_position = masked_sentence.index('[MASK]') + 1 # plus one for [CLS]

            sequence_id = tokenizer.encode(question)

            # start_time and end_time are strings, convert them to float
            
            duration = raw_data[video_id]["duration"]
            video_feature_len = video_features[video_id]['c3d_features'].shape[0]
            start_frame = math.floor(float(start_time) / duration * video_feature_len)
            end_frame  = math.floor(float(end_time) / duration * video_feature_len)

            feature_np = video_features[video_id]['c3d_features'][start_frame:end_frame+1]
            
            if feature_np.shape[0] > 200:
                feature = np.zeros((200, feature_np.shape[1]))
                for i in range(200):
                    feature[i] = feature_np[round(i * (feature_np.shape[0]-1)/199)]
            else:
                feature = feature_np
            
            feature = torch.tensor(feature, dtype=torch.float)

            data.append((sequence_id, feature, worker_answers, mask_position, standard_answer, extended_answers))

    with open(f'{csvData}.pkl', 'wb') as f:
        pickle.dump(data, f)