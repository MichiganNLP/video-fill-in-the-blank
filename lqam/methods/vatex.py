import pickle
from typing import Sequence

import torch
from overrides import overrides
from torch.utils.data import Dataset

DATA_TYPE = Sequence[str, torch.Tensor, str]


class VatexDataset(Dataset):
    """
    Args:
        data_path: input pickle data path

    Returns:
        [masked caption, video I3D features, labels]
    """

    def __init__(self, data_path: str) -> None:
        self.data = self.preprocess_data(data_path)

    @staticmethod
    def preprocess_data(data_path: str) -> Sequence[DATA_TYPE]:
        with open(data_path, "rb") as file:
            raw_data = pickle.load(file)

        data = []
        for d in raw_data:
            question = d[2].lower()
            label = f"<extra_id_0> {d[3]} <extra_id_1>"
            videoFeature = d[4]

            data.append([question, videoFeature, label])

        return data

    @overrides
    def __len__(self) -> int:
        return len(self.data)

    @overrides
    def __getitem__(self, idx) -> DATA_TYPE:
        return self.data[idx]
