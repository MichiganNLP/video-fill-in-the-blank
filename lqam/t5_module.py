from typing import Any, Mapping, Optional

import pytorch_lightning as pl
import torch
from overrides import overrides
from transformers import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import Seq2SeqLMOutput

from lqam.data_module import TYPE_BATCH
from lqam.decoder_utils import compute_label_prob
from lqam.metrics import AlmostExactMatchAccuracy
from lqam.t5_format_processing import compute_blank_map


# Some things were copied from https://github.com/huggingface/transformers/blob/8062fa6/examples/rag/finetune_rag.py#L94
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
        self.extra_id_1 = self.tokenizer.convert_tokens_to_ids(["<extra_id_1>"])[0]

    @overrides
    def on_epoch_start(self) -> None:
        self.accuracy.reset()

    @overrides
    def forward(self, masked_caption_ids: torch.Tensor, label_ids: Optional[torch.Tensor] = None,
                **kwargs) -> Seq2SeqLMOutput:
        return self.t5_pretrained_model(masked_caption_ids, labels=label_ids, **kwargs)

    def _step(self, masked_caption_ids: torch.Tensor, label_ids: torch.Tensor, **_kwargs) -> torch.Tensor:
        return self(masked_caption_ids, label_ids)["loss"]

    def training_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> torch.Tensor:
        return self._step(**batch)

    def _generative_step(self, masked_caption_ids: torch.Tensor, label_ids: torch.Tensor, masked_caption: str,
                         label: str, **_kwargs) -> None:
        self.write_prediction("masked_caption", masked_caption)
        self.write_prediction("ground_truth", label)

        extra_id_0s = torch.ones(len(label_ids), 1, dtype=label_ids.dtype, device=label_ids.device) * self.extra_id_0
        extra_id_1s = torch.ones(len(label_ids), 1, dtype=label_ids.dtype, device=label_ids.device) * self.extra_id_1
        label_ids = torch.cat((extra_id_0s, label_ids, extra_id_1s), dim=1)

        label_output = self(masked_caption_ids, label_ids)
        self.log("loss", label_output["loss"], prog_bar=True)

        # We ignore the EOS token as it's about the EOS of the generation stream, not the end of the blank.
        # The end of the blank is marked by the next extra token.
        # The end of the generation stream was defined on how the model was trained.
        # It's irrelevant when computing the joint probability.
        label_prob = compute_label_prob(label_output["logits"], label_ids,
                                        pad_token_id=self.t5_pretrained_model.config.pad_token_id,
                                        eos_token_id=self.t5_pretrained_model.config.eos_token_id)
        self.write_prediction("ground_truth_prob", label_prob)

        generated_ids = self.t5_pretrained_model.generate(masked_caption_ids, **self.generate_kwargs)
        # TODO: just get the first blank result?
        #   ideally we shouldn't clean it because it should bea already generated until the first blank.
        generated = self.tokenizer.batch_decode(
            blank_map_instance[self.extra_id_0]
            for blank_map_instance in compute_blank_map(generated_ids, self.tokenizer, masked_caption_ids))
        self.write_prediction("generated", generated)

        clean_generated_ids = self.tokenizer(["<extra_id_0> " + generated_instance + " <extra_id_1>"
                                              for generated_instance in generated], return_tensors="pt",
                                             truncation=True,
                                             padding="longest")["input_ids"].to(generated_ids.device)

        # TODO: optimize encoder part?
        generated_output = self(masked_caption_ids, clean_generated_ids)
        generated_prob = compute_label_prob(generated_output["logits"], clean_generated_ids,
                                            pad_token_id=self.t5_pretrained_model.config.pad_token_id,
                                            eos_token_id=self.t5_pretrained_model.config.eos_token_id)
        self.write_prediction("generated_prob", generated_prob)

        accuracy = self.accuracy(generated, label)
        self.log("accuracy", accuracy, prog_bar=True)

    @overrides
    def validation_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> None:
        self._generative_step(**batch)

    @overrides
    def test_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> None:
        self._generative_step(**batch)
