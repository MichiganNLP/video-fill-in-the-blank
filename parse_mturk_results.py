import csv
import h5py
import math
import torch
from transformers import BertTokenizer, BertModel

folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
videoFeatures = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
csvData = f"{folder}/val1_50_mturk_appr_answers.csv"

data = []

def getVideoFeatures(key, startFrame, endFrame, videoFeatures):
        feature_np = videoFeatures[key]['c3d_features'][startFrame:endFrame+1]
        
        if feature_np.shape[0] > 200:
            feature = np.zeros((200, feature_np.shape[1]))
            for i in range(200):
                feature[i] = feature_np[round(i * (feature_np.shape[0]-1)/199)]
        else:
            feature = feature_np

        return torch.tensor(feature, dtype=torch.float)

with open(csvData) as csvfile:
    reader = csv.reader(csvfile)
    isHead = True
    for row in reader:
        if isHead:
            isHead = False
            continue
        video_id, question, start_time, end_time, _, standard_answer, worker_answers = row
        
        # original worker answers is a list-like string, convert it to a real list        
        worker_answers = worker_answers.strip(']\'\'[').split('\', \'') 
        
        extended_answers = list(set([standard_answer] + worker_answers))
        masked_sentence = tokenizer.tokenize(question)
        mask_position = masked_sentence.index('[MASK]') + 1 # plus one for [CLS]

        sequence_id = tokenizer.encode(question)

        # start_time and end_time are strings, convert them to float
        start_frame = math.floor(float(start_time) * 2)
        end_frame = math.ceil(float(end_time) * 2)

        video_features = getVideoFeatures(video_id, start_frame, end_frame, videoFeatures)

        data.append(masked_sentence, video_features, extended_answers, mask_position)

with open('val_mturk.pkl', 'wb') as f:
    pickle.dump(data, f)