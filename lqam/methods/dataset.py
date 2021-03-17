import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from lqam.util.file_utils import cached_path

URL_DATA_TEST = "https://www.dropbox.com/s/2nr7kooprjti975/test.json?dl=1"
URL_DATA_VAL = "https://www.dropbox.com/s/t1dpotaz2sjjtxk/val.json?dl=1"
URL_DATA_TRAIN = "https://www.dropbox.com/s/lc3e1ave94hz9tu/train.json?dl=1"

URL_VAL_LABEL_CATEGORIES = "https://www.dropbox.com/s/3zxz9jtivg7oedr/val_label_categories.tsv?dl=1"
URL_TEST_LABEL_CATEGORIES = "https://www.dropbox.com/s/77koxiu59q2w0vl/test_label_categories.tsv?dl=1"

N_CATEGORIES = 11  # 10 plus the error one.

TYPE_BATCH = MutableMapping[str, Any]


def _load_label_categories_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(cached_path(path), sep="\t")


def load_label_categories() -> Mapping[str, int]:
    val_cat_df = _load_label_categories_from_path(URL_VAL_LABEL_CATEGORIES)
    test_cat_df = _load_label_categories_from_path(URL_TEST_LABEL_CATEGORIES)
    return {row["video_id"]: row["category"] for _, row in itertools.chain(val_cat_df.iterrows(),
                                                                           test_cat_df.iterrows())}


# From https://stackoverflow.com/a/53403392/1165181
# There's also one in https://github.com/allenai/allennlp/blob/4535f5c/allennlp/nn/util.py#L119
def get_mask_from_sequence_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_length = lengths.max()
    return torch.arange(max_length).expand(len(lengths), max_length) < lengths.unsqueeze(1)  # noqa


class QGenDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Optional[PreTrainedTokenizerBase] = None, t5_format: bool = True,
                 visual_data_dir: Optional[str] = None) -> None:
        super().__init__()
        with open(cached_path(data_path)) as file:
            self.instances = json.load(file)
        self.tokenizer = tokenizer
        self.t5_format = t5_format
        self.visual_data_dir = Path(visual_data_dir) if visual_data_dir else None

    def __getitem__(self, i: int) -> TYPE_BATCH:
        instance = self.instances[i]

        output = {
            "masked_caption": instance["masked_caption"],
            "label": instance["label"],
            "video_id": instance["video_id"],
            "video_start_time": instance["video_start_time"],
            "video_end_time": instance["video_end_time"],
        }

        if "additional_answers" in instance:
            output["additional_answers"] = instance["additional_answers"]

        if self.visual_data_dir:
            video_file_name = (f"{instance['video_id']}_{instance['video_start_time']:06d}"
                               f"_{instance['video_end_time']:06d}.npy")
            output["visual"] = torch.from_numpy(np.load(self.visual_data_dir / video_file_name)).squeeze(0)  # noqa

        return output

    def __len__(self) -> int:
        return len(self.instances)

    def collate_fn(self, instances: Iterable[TYPE_BATCH]) -> TYPE_BATCH:
        keys = next(iter(instances), {})
        batch = {k: [instance[k] for instance in instances] for k in keys}

        for k in ["masked_caption", "label"]:
            stack = batch[k]

            if self.tokenizer:
                if self.t5_format:
                    if k == "label":
                        to_tokenize = [f"<extra_id_0> {s} <extra_id_1>" for s in stack]
                    elif k == "masked_caption":
                        to_tokenize = [s.replace("_____", "<extra_id_0>") for s in stack]
                    else:
                        to_tokenize = stack
                else:
                    to_tokenize = stack

                # We tokenize in batches, in parallel. Probably there's a little gain than each worker tokenizing
                # separately each item in a batch because the padding is known a priori and there may be other parallel
                # optimizations. And it's more elegant. Still, it's likely marginal. Though now the workers aren't
                # serial anymore, so we shouldn't use as many workers as CPU cores but just a small number so the
                # compute devices aren't starving but not large so they never compete a lot with each other (esp. at the
                # beginning, where the pipeline of workers is starting).
                tokenization_output = self.tokenizer(to_tokenize, padding="longest", truncation=True,
                                                     return_tensors="pt")
                batch[f"{k}_ids"] = tokenization_output["input_ids"]
                batch[f"{k}_attention_mask"] = tokenization_output["attention_mask"]

        if "visual" in keys:
            visual_list = batch["visual"]
            batch["visual"] = pad_sequence(visual_list, batch_first=True)

            lengths = torch.as_tensor([visual_instance.size(0) for visual_instance in visual_list])
            batch["visual_attention_mask"] = get_mask_from_sequence_lengths(lengths)

        return batch


class QGenDataModule(pl.LightningDataModule):  # noqa
    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase] = None, batch_size: Optional[int] = 32,
                 eval_batch_size: Optional[int] = None, num_workers: int = 0, train_data_path: str = URL_DATA_TRAIN,
                 val_data_path: str = URL_DATA_VAL, test_data_path: str = URL_DATA_TEST,
                 visual_data_dir: Optional[str] = None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.visual_data_dir = visual_data_dir

    def _dataloader(self, data_path: str, batch_size: int, train: bool) -> DataLoader:
        dataset = QGenDataset(data_path, tokenizer=self.tokenizer, visual_data_dir=self.visual_data_dir)
        # TODO: bucket-batching could make training faster, and consume less memory.
        return DataLoader(dataset, shuffle=train, batch_size=batch_size, num_workers=self.num_workers,
                          pin_memory=True, collate_fn=None if batch_size is None else dataset.collate_fn)

    @overrides
    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_data_path, batch_size=self.batch_size, train=True)

    @overrides
    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_data_path, batch_size=self.eval_batch_size, train=False)

    @overrides
    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_data_path, batch_size=self.eval_batch_size, train=False)
