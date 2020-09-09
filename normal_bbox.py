# This file normalize all bbox with the width/height of the original images
import os
from PIL import Image
import pickle
import torch

video_folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/5fps_Videos/"
pickle_folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/latest_data/multimodal_model/object_detection_features/"

for video_data in os.listdir(pickle_folder):
    video = video_data[:-4]
    with Image.open(f'{video_folder}{video}/000001.jpg') as image:
        w, h = image.size
    with open(f'{pickle_folder}{video_data}', 'rb') as f:
        video_features = pickle.load(f)
    
    for feature in video_features:
        boxes = feature[0][0]
        for box in boxes:
            box[0] /= float(w)
            box[1] /= float(h)
            box[2] /= float(w)
            box[3] /= float(h)
    
    with open(f'{pickle_folder}{video_data}', 'wb') as f:
        pickle.dump(video_features, f)