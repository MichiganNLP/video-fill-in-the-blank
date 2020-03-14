from torch.utils.data import Dataset
import torch
import pickle

class ActivityNetCaptionDataset(Dataset):

    def __init__(self, data):
        """
        Args:
            data: data file name
        Output data structure:
            masked sentence
            video feature
            label
            mask position
            video url
        """
        with open(data, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 