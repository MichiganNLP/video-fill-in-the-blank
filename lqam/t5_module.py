from typing import Any, Mapping, Optional

import pytorch_lightning as pl
import torch
from overrides import overrides
from transformers import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import Seq2SeqLMOutput

from lqam.data_module import TYPE_BATCH
from lqam.metrics import AlmostExactMatchAccuracy
from lqam.t5_format_processing import compute_blank_map


# Many things were copied from https://github.com/huggingface/transformers/blob/8062fa6/examples/rag/finetune_rag.py#L94
class T5FillerModel(pl.LightningModule):
    def __init__(self, t5_like_pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                 generate_kwargs: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        # TODO: hparams
        # The model doesn't necessarily use T5 classes (e.g., `T5PreTrainedModel`).
        # It just needs to be pretrained like T5 and support conditional generation.
        assert isinstance(t5_like_pretrained_model, tuple(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.values()))
        self.t5_pretrained_model = t5_like_pretrained_model
        self.tokenizer = tokenizer
        self.accuracy = AlmostExactMatchAccuracy()
        self.generate_kwargs = generate_kwargs or {}

        self.extra_id_0 = self.tokenizer.convert_tokens_to_ids(["<extra_id_0>"])[0]

    @overrides
    def on_epoch_start(self) -> None:
        self.accuracy.reset()

    @overrides
    def forward(self, masked_caption_ids: torch.Tensor, label_ids: Optional[torch.Tensor] = None) -> Seq2SeqLMOutput:
        return self.t5_pretrained_model(masked_caption_ids, labels=label_ids)

    def _step(self, masked_caption_ids: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
        return self(masked_caption_ids, label_ids)["loss"]

    def _generative_step(self, batch: TYPE_BATCH) -> Mapping[str, torch.Tensor]:
        masked_caption_ids = batch["masked_caption_ids"]
        label_ids = batch["label_ids"]
        label = batch["label"]

        generated_ids = self.t5_pretrained_model.generate(masked_caption_ids, **self.generate_kwargs)
        # Use `decode` directly here as it's not a batch an we don't need to skip special tokens.
        generated = [{self.tokenizer.decode(k): self.tokenizer.decode(v) for k, v in blank_map_instance.items()}
                     for blank_map_instance in compute_blank_map(generated_ids, self.tokenizer, masked_caption_ids)]
        generated = [generated_instance["<extra_id_0>"] for generated_instance in generated]  # FIXME

        return {
            "loss": self._step(masked_caption_ids, label_ids),
            "accuracy": self.accuracy(generated, label),
            "masked_caption": batch["masked_caption"],
            "ground_truth": label,
            "generated": generated,
        }

    @overrides
    def validation_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> Mapping[str, torch.Tensor]:
        return self._generative_step(batch)

    @overrides
    def test_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> Mapping[str, torch.Tensor]:
        return self._generative_step(batch)
