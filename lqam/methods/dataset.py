from pathlib import Path
from typing import Any, Iterable, MutableMapping, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from lqam.util.file_utils import cached_path

URL_DATA_TEST = "https://drive.google.com/uc?id=1h-8ADZJDr32QgZMClQ6J1mvMWQY0Ahzx&export=download"
URL_DATA_VAL = "https://drive.google.com/uc?id=1Fv5Yf79guD-95yNNGpFr-GHUMrNc-gSv&export=download"
URL_DATA_TRAIN = "https://drive.google.com/uc?id=1BureM8nfvmgoHxaZeVWeUpYTuTrX_Kcx&export=download"

TYPE_BATCH = MutableMapping[str, Any]


# From https://stackoverflow.com/a/53403392/1165181
# There's also one in https://github.com/allenai/allennlp/blob/4535f5c/allennlp/nn/util.py#L119
def get_mask_from_sequence_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_length = lengths.max()
    return torch.arange(max_length).expand(len(lengths), max_length) < lengths.unsqueeze(1)  # noqa


class QGenDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerBase, t5_format: bool = True,
                 visual_data_dir: Optional[str] = None) -> None:
        super().__init__()
        self.df = pd.read_csv(cached_path(data_path))
        self.tokenizer = tokenizer
        self.t5_format = t5_format
        self.visual_data_dir = Path(visual_data_dir) if visual_data_dir else None

    def __getitem__(self, i: int) -> TYPE_BATCH:
        row = self.df.iloc[i]

        output = {
            "masked_caption": row["masked_caption"],
            "label": row["label"],
        }

        if self.visual_data_dir:
            video_file_name = f"{row['video_id']}_{row['video_start_time']:06d}_{row['video_end_time']:06d}.npy"
            output["visual"] = torch.from_numpy(np.load(self.visual_data_dir / video_file_name)).squeeze(0)  # noqa

        return output

    def __len__(self) -> int:
        return len(self.df)

    def collate_fn(self, instances: Iterable[TYPE_BATCH]) -> TYPE_BATCH:
        batch = {}

        for k in ["masked_caption", "label"]:
            stack = [instance[k] for instance in instances]
            batch[k] = stack

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
            # optimizations. And it's more elegant. Still, it's likely marginal. Though now the workers aren't serial
            # anymore, so we shouldn't use as many workers as CPU cores but just a small number so the compute
            # devices aren't starving but not large so they never compete a lot with each other (esp. at the
            # beginning, where the pipeline of workers is starting).
            tokenization_output = self.tokenizer(to_tokenize, padding="longest", truncation=True, return_tensors="pt")
            batch[f"{k}_ids"] = tokenization_output["input_ids"]
            batch[f"{k}_attention_mask"] = tokenization_output["attention_mask"]

        if "visual" in next(iter(instances), {}):
            visual_list = [instance["visual"] for instance in instances]
            batch["visual"] = pad_sequence(visual_list, batch_first=True)

            lengths = torch.as_tensor([visual_instance.size(0) for visual_instance in visual_list])
            batch["visual_attention_mask"] = get_mask_from_sequence_lengths(lengths)

        return batch


class QGenDataModule(pl.LightningDataModule):  # noqa
    def __init__(self, tokenizer: PreTrainedTokenizerBase, batch_size: int = 32, eval_batch_size: Optional[int] = None,
                 num_workers: int = 0, train_data_path: str = URL_DATA_TRAIN, val_data_path: str = URL_DATA_VAL,
                 test_data_path: str = URL_DATA_TEST, visual_data_dir: Optional[str] = None):
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
                          pin_memory=True, collate_fn=dataset.collate_fn)

    @overrides
    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_data_path, batch_size=self.batch_size, train=True)

    @overrides
    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_data_path, batch_size=self.eval_batch_size, train=False)

    @overrides
    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_data_path, batch_size=self.eval_batch_size, train=False)
