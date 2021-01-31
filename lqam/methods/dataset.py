import pickle
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd
import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase
import numpy as np
import os
import torch

import pickle

from lqam.util.file_utils import cached_path

URL_DATA_TEST = "https://drive.google.com/uc?id=1h-8ADZJDr32QgZMClQ6J1mvMWQY0Ahzx&export=download"
URL_DATA_VAL = "https://drive.google.com/uc?id=1Fv5Yf79guD-95yNNGpFr-GHUMrNc-gSv&export=download"
URL_DATA_TRAIN = "https://drive.google.com/uc?id=1BureM8nfvmgoHxaZeVWeUpYTuTrX_Kcx&export=download"

TYPE_BATCH = Mapping[str, Any]


class QGenDataset(Dataset):
    def __init__(self, data_path: str, visual_data_path:str, tokenizer: PreTrainedTokenizerBase, return_visual: bool = False,
                 t5_format: bool = True) -> None:
        super().__init__()
            
        self.data = pd.read_csv(cached_path(data_path))

        self.visual_data_path = visual_data_path
        self.tokenizer = tokenizer
        self.return_visual = return_visual
        self.t5_format = t5_format

    def __getitem__(self, i: int) -> TYPE_BATCH:
        row = self.data.iloc[i]
        if not self.return_visual:
            # The masked caption is already in T5 format: "<extra_id_0>" is the blank name.
            return {
                "masked_caption": row.get("masked caption", row.get("masked_caption")).replace("<extra_id_0>", "_____"),
                "label": row["label"],
            }
        else:
            video_id = row['video_id']
            video_feature = torch.LongTensor(np.load(os.path.join(self.visual_data_path, video_id + '.npy'))).squeeze(0)
            return {
                "masked_caption": row.get("masked caption", row.get("masked_caption")).replace("<extra_id_0>", "_____"),
                "label": row["label"],
                'visual': video_feature
            }

    def __len__(self) -> int:
        return len(self.data)

    def collate_fn(self, instances: Iterable[TYPE_BATCH]) -> TYPE_BATCH:
        batch = {}
        for k in ["masked_caption", "label"]:
            stack = [instance[k] for instance in instances]
            batch[k] = stack

            if self.t5_format and k == "label":
                to_tokenize = [f"<extra_id_0> {s} <extra_id_1>" for s in stack]
            else:
                to_tokenize = stack

            # We tokenize in batches, in parallel. Probably there's a little gain than each worker tokenizing
            # separately each item in a batch because the padding is known a priori and there may be other parallel
            # optimizations. And it's more elegant. Still, it's likely marginal. Though now the workers aren't serial
            # anymore, so we shouldn't use as many workers as CPU cores but just a small number so the compute
            # devices aren't starving but not large so they never compete a lot with each other (esp. at the
            # beginning, where the pipeline of workers is starting).
            tokenization_output = self.tokenizer(to_tokenize, padding="longest", truncation=True, return_tensors="pt")
            batch[f"{k}_ids"] = tokenization_output["input_ids"]
            batch[f"{k}_attention_mask"] = tokenization_output["attention_mask"]
        return batch

    def collate_fn_multi_modal(self, batch: Sequence[Sequence[Any]]) -> TYPE_BATCH:
        batch_size = len(batch)
        masked_captions = []
        video_features = []
        label_list = []

        max_video_len = 0
        visual_size = batch[0]["visual"].shape[1]
        for i in range(batch_size):
            data = batch[i]
            masked_captions.append(data["masked_caption"])
            video_features.append(data["visual"])
            label_list.append(data["label"])

            total_video_len = data["visual"].shape[0]

            if total_video_len > max_video_len:
                max_video_len = total_video_len

        label_with_special_tokens = [f"<extra_id_0> {label} <extra_id_1>" for label in label_list]
        text_batch = self.tokenizer.prepare_seq2seq_batch(src_texts=masked_captions, tgt_texts=label_with_special_tokens,
                                                          padding=True, return_tensors="pt")
        text_tensor = text_batch.input_ids
        text_attention_mask = text_batch.attention_mask
        labels = text_batch.labels

        video_tensor = torch.zeros(batch_size, max_video_len, visual_size, dtype=torch.float)

        video_attention_mask = torch.zeros(batch_size, max_video_len, dtype=torch.long)

        for i in range(batch_size):
            video = video_features[i]
            video_len = len(video)

            # The input to the transformer is gonna be:
            # t_1 ... t_n pad ... pad </s> v_1 ... v_m pad ... pad

            video_len = video.shape[0]
            video_tensor[i, :video_len] = video
            video_attention_mask[i, :video_len] = True

        attention_mask = torch.cat([text_attention_mask, video_attention_mask], 1)

        return {"masked_caption_ids": text_tensor, "visual": video_tensor, "label_ids": labels,
                "masked_caption": masked_captions,
                "label": label_list, 'masked_caption_attention_mask': attention_mask}


class QGenDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, batch_size: int = 32, eval_batch_size: Optional[int] = None,
                 num_workers: int = 0, hasVisual=False, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.hasVisual = hasVisual

    def _dataloader(self, data_path: str, visual_data_path:str, batch_size: int, train: bool) -> DataLoader:
        dataset = QGenDataset(data_path, visual_data_path, tokenizer=self.tokenizer, return_visual=self.hasVisual)
        # TODO: bucket-batching could make training faster, and consume less memory.
        if self.hasVisual:
            return DataLoader(dataset, shuffle=train, batch_size=batch_size, num_workers=self.num_workers,
                              pin_memory=True, collate_fn=dataset.collate_fn_multi_modal)
        else:
            return DataLoader(dataset, shuffle=train, batch_size=batch_size, num_workers=self.num_workers,
                              pin_memory=True, collate_fn=dataset.collate_fn)

    @overrides
    def train_dataloader(self, data_path: str = URL_DATA_TRAIN, visual_data_path: Optional[str] = None) -> DataLoader:
        return self._dataloader(data_path, visual_data_path=visual_data_path, batch_size=self.batch_size, train=True)

    @overrides
    def val_dataloader(self, data_path: str = URL_DATA_VAL, visual_data_path: Optional[str] = None) -> DataLoader:
        return self._dataloader(data_path, visual_data_path=visual_data_path, batch_size=self.eval_batch_size, train=False)

    @overrides
    def test_dataloader(self, data_path: str = URL_DATA_TEST, visual_data_path: Optional[str] = None) -> DataLoader:
        return self._dataloader(data_path, visual_data_path=visual_data_path, batch_size=self.eval_batch_size, train=False)
