from data_loader_mulitmodal import ActivityNetCaptionDataset
import h5py

PATH = 'Checkpoint'
folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"

videoFeatures = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
trainTextFile = f"{folder}/train.json"

ActivityNetCaptionDataset(trainTextFile, videoFeatures)