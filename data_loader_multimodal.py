import pickle

from torch.utils.data import Dataset


class ActivityNetCaptionsDataset(Dataset):
    def __init__(self, pickle_path: str) -> None:
        """
        Args:
            pickle_path: data file name
        Output data structure:
            masked sentence
            video feature
            label
            mask position
            video url
        """
        with open(pickle_path, 'rb') as file:
            self.data_dict = pickle.load(file)

    def __len__(self) -> int:
        return len(self.data_dict)

    def __getitem__(self, idx):
        return self.data_dict[idx]
