import csv
import numpy as np
import os
import torch
import pickle

DATA_FILE_FOLDER = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/latest_data/multimodal_model/VATEX"
DIV_NAME = "train"

data = []

with open(f"{DATA_FILE_FOLDER}/{DIV_NAME}.csv", 'r') as csvFile:
    reader = csv.reader(csvFile, delimiter=",")
    isHead = True
    for row in reader:
        if isHead:
            isHead = False
            continue
        videoID = row[0]
        videoFilename = videoID + ".npy"
        videoPath = os.path.join(DATA_FILE_FOLDER, "val", videoFilename)
        videoFeature_np = np.load(videoPath)
        videoFeature_pt = torch.tensor(videoFeature_np).squeeze(0)
    # videoID, caption, masked caption, label, video I3D features
        data.append(*row, videoFeature_pt)

with open(f"{DATA_FILE_FOLDER}/{DIV_NAME}.pkl", 'wb') as f:
    pickle.dump(data, f)