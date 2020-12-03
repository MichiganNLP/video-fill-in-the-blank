from torch.utils.data import Dataset
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pickle

class VATEX_Dataset(Dataset):

    def __init__(self, data_path):
        """
        Args:
            data_path: input pickle data path
        Output data structure:
            [masked caption, video I3D features, labels]
        """
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.data = self.preprocessData(data_path)

    def preprocessData(self, data_path):
        data = []
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)

        for d in raw_data:
            question = d[2]
            label = d[3]
            videoFeature = d[4]
            
            data.append([question, videoFeature, label])

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return data[idx]
        