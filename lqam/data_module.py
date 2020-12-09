from typing import Any, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from lqam.file_utils import cached_path


class QGenDataModule(pl.LightningDataModule):  # noqa
    def __init__(self, data_path: str, max_data_points: int = 100) -> None:
        super().__init__()
        self.df = pd.read_csv(cached_path(data_path))[:max_data_points]

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def val_dataloader(self) -> DataLoader:
        pass

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass
