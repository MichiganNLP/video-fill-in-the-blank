from typing import Any, Iterable, Literal, Mapping, Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase, TensorType
from transformers.tokenization_utils_base import PaddingStrategy

from lqam.file_utils import cached_path

TYPE_BATCH = Mapping[str, Any]


# From https://github.com/huggingface/transformers/blob/8062fa6/examples/rag/utils_rag.py#L35
def trim_batch(input_ids: torch.Tensor, pad_token_id: int,
               attention_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Remove columns that are populated exclusively by `pad_token_id`."""
    keep_column_mask = (input_ids != pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask]


class QGenDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerBase, return_visual: bool = False) -> None:
        super().__init__()
        self.df = pd.read_csv(cached_path(data_path))
        self.tokenizer = tokenizer
        self.return_visual = return_visual

    def _tokenize(self, s: str, truncation: bool = True,
                  return_tensors: Optional[Union[Literal["pt"], TensorType]] = "pt",
                  padding: Optional[Union[bool, Literal["longest", "max_length", "do_not_pad"],
                                          PaddingStrategy]] = "max_length",
                  **kwargs) -> torch.Tensor:
        return self.tokenizer(s, padding=padding, truncation=truncation, return_tensors=return_tensors,
                              **kwargs)["input_ids"]

    def __getitem__(self, i: int) -> TYPE_BATCH:
        row = self.df.iloc[i]
        # The masked caption is already in T5 format: "<extra_id_0>" is the blank name.
        masked_caption = row["masked caption"]
        label = row["label"]
        # TODO: return the visual features if `self.return_visual`.
        return {
            "masked_caption": masked_caption,
            "label": label,
            "masked_caption_ids": self._tokenize(masked_caption),
            "label_ids": self._tokenize(label),
        }

    def __len__(self) -> int:
        return len(self.df)

    def collate_fn(self, instances: Iterable[TYPE_BATCH]) -> TYPE_BATCH:
        batch = {k: trim_batch(torch.stack([x[k] for x in instances]), self.tokenizer.pad_token_id)
                 for k in ["masked_caption_ids", "label_ids"]}
        for k in ["masked_caption", "label"]:
            batch[k] = [x[k] for x in instances]
        return batch


class QGenDataModule(pl.LightningDataModule):  # noqa
    def __init__(self, tokenizer: PreTrainedTokenizerBase, batch_size: int = 32, eval_batch_size: Optional[int] = None,
                 num_workers: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size

    def _dataloader(self, data_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = QGenDataset(data_path, tokenizer=self.tokenizer)
        # TODO: bucket batching
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_workers,
                          pin_memory=True, collate_fn=dataset.collate_fn)

    @overrides
    def train_dataloader(self, data_path: str = "https://drive.google.com/uc?id=1-5nFmc0bkNUn7V4wMB6j3mOCksX18Lr0"
                                                "&export=download") -> DataLoader:
        return self._dataloader(data_path, batch_size=self.batch_size, shuffle=True)

    @overrides
    def val_dataloader(self, data_path: str = "https://drive.google.com/uc?id=1-JRsjFzP3Qmjti_w8ILV06msXjw4OXoB"
                                              "&export=download") -> DataLoader:
        return self._dataloader(data_path, batch_size=self.eval_batch_size)

    @overrides
    def test_dataloader(self, data_path: str = "https://drive.google.com/uc?id=1-5rnoxSGkf9UyO9xhhwkf7tuElyXG4Yn"
                                               "&export=download") -> DataLoader:
        return self._dataloader(data_path, batch_size=self.eval_batch_size)
