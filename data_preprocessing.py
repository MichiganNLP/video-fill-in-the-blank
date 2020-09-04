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
# trainTextFile = "Train.tsv"
# valTextFile = "Validation.tsv"
# testTextFile = "Test.tsv"
trainTextFile = "TrainSubset.csv"
valTextFile = "ValidationSubset.csv"
testTextFile = "TestSubset.csv"
durationFile = f"{folder}/latest_data/multimodal_model/video_duration.pkl"

ActivityNetCaptionDataset(trainTextFile, videoFeatures, durationFile, 'train', isTrain=True)
print("train done")
# ActivityNetCaptionDataset(valTextFile, videoFeatures, durationFile, 'val', isTrain=False)
# print("val done")
# ActivityNetCaptionDataset(testTextFile, videoFeatures, durationFile, 'test', isTrain=False)
# print("test done")