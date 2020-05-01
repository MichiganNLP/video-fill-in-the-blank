from data_loader_multimodal_old import ActivityNetCaptionDataset
import h5py
import random
import torch
import numpy as np

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

PATH = 'Checkpoint'
folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"

videoFeatures = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
trainTextFile = f"{folder}/val_1.json"

ActivityNetCaptionDataset(trainTextFile, videoFeatures, isTrain=False)
