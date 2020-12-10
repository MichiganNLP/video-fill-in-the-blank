from typing import Any, Mapping, Optional

import pytorch_lightning as pl
import torch
from transformers import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from lqam.data_module import TYPE_BATCH


class T5Filler(pl.LightningModule):
    def __init__(self, t5_like_pretrained_model: PreTrainedModel,
                 generate_kwargs: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        # TODO: hparams
        # The model doesn't necessarily use T5 classes (e.g., `T5PreTrainedModel`).
        # It just needs to be pretrained like T5 and support conditional generation.
        assert isinstance(t5_like_pretrained_model, tuple(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.values()))
        self.t5_pretrained_model = t5_like_pretrained_model
        self.generate_kwargs = generate_kwargs or {}

    def forward(self, masked_caption_ids: torch.Tensor, label_ids: Optional[torch.Tensor] = None) -> Seq2SeqLMOutput:
        return self.t5_pretrained_model(masked_caption_ids, labels=label_ids, **self.generate_kwargs)

    def _step(self, batch: TYPE_BATCH) -> float:
        outputs = self(**batch)
        return outputs["loss"]

    def _generative_step(self, batch: TYPE_BATCH) -> float:
        return self._step(batch)

    def validation_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> float:
        return self._generative_step(batch)

    def test_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> float:
        return self._generative_step(batch)
