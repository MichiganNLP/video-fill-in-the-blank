from typing import Optional

import pytorch_lightning as pl
import torch
from transformers import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, PreTrainedModel

from lqam.data_module import TYPE_BATCH


class T5Filler(pl.LightningModule):
    def __init__(self, t5_like_pretrained_model: PreTrainedModel) -> None:
        super().__init__()
        # The model doesn't necessarily use T5 classes (e.g., `T5PreTrainedModel`).
        # It just needs to be pretrained like T5 and support conditional generation.
        assert isinstance(t5_like_pretrained_model, tuple(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.values()))
        self.t5_pretrained_model = t5_like_pretrained_model

    def forward(self, masked_caption_ids: torch.Tensor, label_ids: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        return self.t5_pretrained_model.generate(masked_caption_ids, **kwargs)

    def _generative_step(self, batch: TYPE_BATCH):
        return self(**batch)

    def validation_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> torch.Tensor:
        return self._generative_step(batch)

    def test_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> torch.Tensor:
        return self._generative_step(batch)
