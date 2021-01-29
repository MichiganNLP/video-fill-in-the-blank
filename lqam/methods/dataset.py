from typing import Any, Iterable, Mapping, Optional

import pandas as pd
import pytorch_lightning as pl
from overrides import overrides
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from lqam.util.file_utils import cached_path

URL_DATA_TEST = "https://drive.google.com/uc?id=1h-8ADZJDr32QgZMClQ6J1mvMWQY0Ahzx&export=download"
URL_DATA_VAL = "https://drive.google.com/uc?id=1Fv5Yf79guD-95yNNGpFr-GHUMrNc-gSv&export=download"
URL_DATA_TRAIN = "https://drive.google.com/uc?id=1BureM8nfvmgoHxaZeVWeUpYTuTrX_Kcx&export=download"

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
        # TODO: return the visual features if `self.return_visual`.
        return {
            # FIXME: `replace` and "masked caption" key should be removed when the dataset files are fixes.
            "masked_caption": row.get("masked caption", row.get("masked_caption")).replace("<extra_id_0>", "_____"),
            "label": row["label"],
        }

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
        return batch


class QGenDataModule(pl.LightningDataModule):  # noqa
    def __init__(self, tokenizer: PreTrainedTokenizerBase, batch_size: int = 32, eval_batch_size: Optional[int] = None,
                 num_workers: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size

    def _dataloader(self, data_path: str, batch_size: int, train: bool) -> DataLoader:
        dataset = QGenDataset(data_path, tokenizer=self.tokenizer)
        # TODO: bucket-batching could make training faster, and consume less memory.
        return DataLoader(dataset, shuffle=train, batch_size=batch_size, num_workers=self.num_workers,
                          pin_memory=True, collate_fn=dataset.collate_fn)

    @overrides
    def train_dataloader(self, data_path: str = URL_DATA_TRAIN) -> DataLoader:
        return self._dataloader(data_path, batch_size=self.batch_size, train=True)

    @overrides
    def val_dataloader(self, data_path: str = URL_DATA_VAL) -> DataLoader:
        return self._dataloader(data_path, batch_size=self.eval_batch_size, train=False)

    @overrides
    def test_dataloader(self, data_path: str = URL_DATA_TEST) -> DataLoader:
        return self._dataloader(data_path, batch_size=self.eval_batch_size, train=False)
