from typing import Any, Mapping, Optional, Sequence

import pytorch_lightning as pl
import torch
from overrides import overrides
from transformers import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import Seq2SeqLMOutput

from lqam.methods.dataset import TYPE_BATCH
from lqam.methods.decoding import compute_label_prob, compute_noun_phrase_indices
from lqam.methods.metrics import AlmostExactMatchAccuracy
from lqam.methods.t5_format_processing import compute_first_blank

import spacy


# Some things were copied from https://github.com/huggingface/transformers/blob/8062fa6/examples/rag/finetune_rag.py#L94
class T5FillerModel(pl.LightningModule):
    def __init__(self, t5_like_pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                 only_noun_phrases: bool = False, generate_kwargs: Optional[Mapping[str, Any]] = None) -> None:
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

        self.only_noun_phrases = only_noun_phrases
        if only_noun_phrases:
            # Note `num_beams` is needed, otherwise the flag doesn't make sense.
            self.spacy_model = spacy.load("en_core_web_lg")
            self.generate_kwargs.setdefault("num_return_sequences", self.generate_kwargs["num_beams"])

        self.all_token_ids = torch.arange(self.tokenizer.vocab_size)
        if self.generate_kwargs.get("num_beams", 1) > 1:
            # We constrain the generation to one blank during beam search. If we don't constrain it, it produces
            # gibberish after generating the first blank value. During beam search, the beams may differ only in
            # gibberish, that's why we want to constrain it. For greedy search it doesn't change anything and it just
            # makes it slower, so we disable it.
            self.generate_kwargs.setdefault("prefix_allowed_tokens_fn", self._prefix_allowed_ids)

    @overrides
    def on_epoch_start(self) -> None:
        self.accuracy.reset()

    @overrides
    def forward(self, masked_caption_ids: torch.Tensor, label_ids: Optional[torch.Tensor] = None,
                **kwargs) -> Seq2SeqLMOutput:
        return self.t5_pretrained_model(masked_caption_ids, labels=label_ids, **kwargs)

    def _step(self, masked_caption_ids: torch.Tensor, label_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self(masked_caption_ids, label_ids, **kwargs)["loss"]

    def training_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> torch.Tensor:
        return self._step(**batch)

    def _prefix_allowed_ids(self, _batch_id: int, input_ids: torch.Tensor) -> Sequence[int]:
        return [self.tokenizer.eos_token_id] if input_ids[-1] == self.extra_id_1 else self.all_token_ids

    def _generative_step(self, masked_caption_ids: torch.Tensor, label_ids: torch.Tensor, masked_caption: str,
                         label: str, **_kwargs) -> None:
        self.write_prediction("masked_caption", masked_caption)
        self.write_prediction("ground_truth", label)

        model_kwargs = {}

        if self.t5_pretrained_model.config.is_encoder_decoder:
            # Compute the encoder part only once.
            # For the `generate` method it doesn't apply because it always computes it. We can't do much w/o
            # implementing our own version.
            encoder = self.t5_pretrained_model.get_encoder()
            model_kwargs["encoder_outputs"] = encoder(masked_caption_ids)

        label_output = self(masked_caption_ids, label_ids, **model_kwargs)
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
        generated = self.tokenizer.batch_decode(
            compute_first_blank(generated_ids, self.t5_pretrained_model.config.decoder_start_token_id,
                                self.extra_id_0, self.extra_id_1))

        if "prefix_allowed_tokens_fn" in self.generate_kwargs:
            # The generation was constrained, so it's already clean. But we need to remove the BoS token used to
            # start the generation. I noticed later that there's some `view(...)` in T5 code that fails due to the
            # inside  stride after doing this. So I added a `clone()`.
            clean_generated_ids = generated_ids[:, 1:].clone()
        else:
            clean_generated_ids = self.tokenizer(
                ["<extra_id_0> " + generated_instance + " <extra_id_1>" for generated_instance in generated],
                return_tensors="pt", truncation=True, padding="longest")["input_ids"].to(generated_ids.device)
            
        if self.only_noun_phrases:
            num_return_sequences = self.generate_kwargs.get("num_return_sequences", 1)
            batch_size = len(generated) // num_return_sequences
            assert len(generated) % num_return_sequences == 0
            
            noun_chunks_indices = compute_noun_phrase_indices(self.spacy_model, generated, batch_size,
                                                              num_return_sequences, generated_ids.device)
            # filter out unqualified sequences, and leave one answer for each instance
            clean_generated_ids = clean_generated_ids[noun_chunks_indices]
            # extract the answers that are either noun phrases with the highest prob or the first answer
            generated = [generated[i.item()] for i in noun_chunks_indices]
            
        generated_output = self(masked_caption_ids, clean_generated_ids, **model_kwargs)
        generated_prob = compute_label_prob(generated_output["logits"], clean_generated_ids,
                                            pad_token_id=self.t5_pretrained_model.config.pad_token_id,
                                            eos_token_id=self.t5_pretrained_model.config.eos_token_id)

        self.write_prediction("generated", generated)
        self.write_prediction("generated_prob", generated_prob)

        accuracy = self.accuracy(generated, label)
        self.log("accuracy_step", accuracy, prog_bar=True)

    @overrides
    def validation_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> None:
        self._generative_step(**batch)

    @overrides
    def test_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> None:
        self._generative_step(**batch)

    def _on_epoch_end(self) -> None:
        self.log("accuracy", self.accuracy.compute(), prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end()

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end()
