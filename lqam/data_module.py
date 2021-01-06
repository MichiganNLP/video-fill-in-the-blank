from typing import Any, Iterable, Mapping, Optional

import pandas as pd
import pytorch_lightning as pl
from overrides import overrides
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from lqam.file_utils import cached_path

TYPE_BATCH = Mapping[str, Any]


class QGenDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerBase, return_visual: bool = False,
                 t5_format: bool = True) -> None:
        super().__init__()
        self.df = pd.read_csv(cached_path(data_path))
        self.tokenizer = tokenizer
        self.return_visual = return_visual
        self.t5_format = t5_format

    def __getitem__(self, i: int) -> TYPE_BATCH:
        row = self.df.iloc[i]
        # The masked caption is already in T5 format: "<extra_id_0>" is the blank name.
        masked_caption = row["masked caption"]
        label = row["label"]
        # TODO: return the visual features if `self.return_visual`.
        return {
            "masked_caption": masked_caption,
            "label": label,
        }

    def __len__(self) -> int:
        return len(self.df)

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
            batch[f"{k}_ids"] = self.tokenizer(to_tokenize, padding="longest", truncation=True,
                                               return_tensors="pt")["input_ids"]
        return batch
    
    def collate_fn_multi_modal(self, batch: Sequence[Sequence[Any]]) -> TYPE_BATCH:
        batch_size = len(batch)
        text_features = []
        video_features = []
        label_list = []

        max_video_len = 0

        for i in range(batch_size):
            data = batch[i]
            text_features.append(data[0])
            video_features.append(data[1])
            label_list.append(data[2])

            if data[1] != None:
                total_video_len = data[1].shape[0]
            else:
                total_video_len = 0
            if total_video_len > max_video_len:
                max_video_len = total_video_len
        
        text_batch = self.tokenizer.prepare_seq2seq_batch(src_texts=text_features,tgt_texts=label_list, padding=True, return_tensors="pt")
        text_tensor = text_batch.input_ids
        text_attention_mask = text_batch.attention_mask
        labels = text_batch.labels

        if self.hparams.enable_visual_features:
            video_tensor = torch.zeros(batch_size, max_video_len, self.hparams.visual_size, dtype=torch.float)
        else:
            video_tensor = None

        if self.hparams.enable_visual_features:
            video_attention_mask = torch.zeros(batch_size, max_video_len, dtype=torch.long)

        for i in range(batch_size):
            video = video_features[i]
            video_len = len(video)

            # The input to the transformer is gonna be:
            # t_1 ... t_n pad ... pad </s> v_1 ... v_m pad ... pad

            if self.hparams.enable_visual_features and video != None:
                video_len = video.shape[0]
                video_tensor[i, :video_len] = video
                video_attention_mask[i, :video_len] = True

        attention_mask = torch.cat([text_attention_mask, video_attention_mask], 1)

        return (text_tensor, video_tensor, attention_mask, labels, label_list)

class QGenDataModule(pl.LightningDataModule):  # noqa
    def __init__(self, tokenizer: PreTrainedTokenizerBase, batch_size: int = 32, eval_batch_size: Optional[int] = None,
                 num_workers: int = 0, hasVisual=False, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.hasVisual = hasVisual

    def _dataloader(self, data_path: str, batch_size: int, train: bool) -> DataLoader:
        dataset = QGenDataset(data_path, tokenizer=self.tokenizer)
        # TODO: bucket-batching could make training faster, and consume less memory.
        if self.hasVisual:
            return DataLoader(dataset, shuffle=train, batch_size=batch_size, num_workers=self.num_workers,
                          pin_memory=True, collate_fn=dataset.collate_fn_multi_modal)
        else:
            return DataLoader(dataset, shuffle=train, batch_size=batch_size, num_workers=self.num_workers,
                          pin_memory=True, collate_fn=dataset.collate_fn)

    @overrides
    def train_dataloader(self, data_path: str = "https://drive.google.com/uc?id=1-5nFmc0bkNUn7V4wMB6j3mOCksX18Lr0"
                                                "&export=download") -> DataLoader:
        return self._dataloader(data_path, batch_size=self.batch_size, train=True)

    @overrides
    def val_dataloader(self, data_path: str = "https://drive.google.com/uc?id=1-JRsjFzP3Qmjti_w8ILV06msXjw4OXoB"
                                              "&export=download") -> DataLoader:
        return self._dataloader(data_path, batch_size=self.eval_batch_size, train=False)

    @overrides
    def test_dataloader(self, data_path: str = "https://drive.google.com/uc?id=1-5rnoxSGkf9UyO9xhhwkf7tuElyXG4Yn"
                                               "&export=download") -> DataLoader:
        return self._dataloader(data_path, batch_size=self.eval_batch_size, train=False)
