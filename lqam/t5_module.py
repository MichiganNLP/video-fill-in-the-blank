from typing import Any, Iterator, Mapping, Optional

import pytorch_lightning as pl
import torch
from transformers import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import Seq2SeqLMOutput

from lqam.data_module import TYPE_BATCH
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
        self.generate_kwargs = generate_kwargs or {}

    def forward(self, masked_caption_ids: torch.Tensor, label_ids: Optional[torch.Tensor] = None) -> Seq2SeqLMOutput:
        return self.t5_pretrained_model(masked_caption_ids, labels=label_ids)

    def _step(self, batch: TYPE_BATCH) -> torch.Tensor:
        return self(**batch)["loss"]

    def _ids_to_clean_text(self, generated_ids: Iterator[int]):
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def _generative_step(self, batch: TYPE_BATCH) -> Mapping[str, torch.Tensor]:
        masked_caption_ids = batch["masked_caption_ids"]

        generated_ids = self.t5_pretrained_model.generate(masked_caption_ids, **self.generate_kwargs)

        generated_tokens = [compute_blank_map(generated_ids_instance, self.tokenizer,
                                              self.tokenizer.convert_ids_to_tokens(
                                                  masked_caption_ids_instance))["<extra_id_0>"]
                            for masked_caption_ids_instance, generated_ids_instance in zip(masked_caption_ids,
                                                                                           generated_ids)]

        return {
            "loss": self._step(batch),
            "masked_caption": self._ids_to_clean_text(masked_caption_ids),  # TODO: also pass as arg.
            "ground_truth": self._ids_to_clean_text(batch["label_ids"]),  # TODO: also pass as arg.
            "generated": [self.tokenizer.convert_tokens_to_string(generated_tokens_instance)
                          for generated_tokens_instance in generated_tokens],  # TODO: make it better.
        }

    def validation_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> Mapping[str, torch.Tensor]:
        return self._generative_step(batch)

    def test_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> Mapping[str, torch.Tensor]:
        return self._generative_step(batch)
