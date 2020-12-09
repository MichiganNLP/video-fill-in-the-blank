from typing import Literal, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lqam.file_utils import cached_path


class QGenDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.df = pd.read_csv(cached_path(data_path))
        self.tokenizer = tokenizer

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[i]
        # The masked caption is already in T5 format: "<extra_id_0>" is the blank name.
        masked_caption_ids = self.tokenizer(row["masked caption"], return_tensors="pt")["input_ids"]
        label_ids = self.tokenizer(row["label"], return_tensors="pt")["input_ids"]
        return masked_caption_ids, label_ids

    def __len__(self) -> int:
        return len(self.df)


class QGenDataModule(pl.LightningDataModule):  # noqa
    def setup(self, stage: Optional[Literal["fit", "test"]] = None) -> None:
        pass

    def val_dataloader(self, data_path: str = "https://drive.google.com/uc?id=1-JRsjFzP3Qmjti_w8ILV06msXjw4OXoB"
                                              "&export=download") -> DataLoader:
        dataset = QGenDataset(data_path, tokenizer=AutoTokenizer.from_pretrained("t5-base"))
        return DataLoader(dataset)
