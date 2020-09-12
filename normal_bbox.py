# This file normalize all bbox with the width/height of the original images
import os
from PIL import Image
import pickle
import torch

video_folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/5fps_Videos/"
pickle_folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/latest_data/multimodal_model/object_detection_features/"

for video_data in os.listdir(pickle_folder):
    with open(f'{pickle_folder}{video_data}', 'rb') as f:
        video_features = pickle.load(f)
    
    new_video_features = []
    for feature in video_features:
        boxes, box_features, scores, labels = feature
        new_video_features.append([[boxes[0].cpu()], [box_features[0].cpu()], [scores[0].cpu()], [labels[0].cpu()]])
    
    with open(f'{pickle_folder}{video_data}', 'wb') as f:
        pickle.dump(new_video_features, f)