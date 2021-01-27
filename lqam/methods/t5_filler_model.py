from typing import Any, Mapping, Optional, Sequence, Union, Iterable, Tuple

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from overrides import overrides
from transformers import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, PreTrainedModel, PreTrainedTokenizerBase, AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from lqam.core.noun_phrases import create_spacy_model
from lqam.methods.dataset import TYPE_BATCH
from lqam.methods.decoding import arg_noun_phrase, compute_answer_prob
from lqam.methods.metrics import AlmostExactMatchAccuracy
from lqam.methods.t5_format_processing import compute_first_blank
from lqam.util.iterable_utils import chunks

# Some things were copied from https://github.com/huggingface/transformers/blob/8062fa6/examples/rag/finetune_rag.py#L94
class T5FillerModel(pl.LightningModule):
    def __init__(self, t5_like_pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, optimizer_args: Mapping[str, Any],
                 only_noun_phrases: bool = False, generate_kwargs: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        # TODO: hparams
        # The model doesn't necessarily use T5 classes (e.g., `T5PreTrainedModel`).
        # It just needs to be pretrained like T5 and support conditional generation.
        assert isinstance(t5_like_pretrained_model, tuple(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.values()))
        self.t5_pretrained_model = t5_like_pretrained_model
        self.tokenizer = tokenizer
        self.accuracy = AlmostExactMatchAccuracy()
        self.optimizer_args = optimizer_args
        self.generate_kwargs = generate_kwargs or {}

        self.generate_kwargs.setdefault("return_dict_in_generate", True)
        self.generate_kwargs.setdefault("output_scores", True)

        # The following are useful to compute the encoder layer output only once.
        self.generate_kwargs.setdefault("output_hidden_states", True)
        self.generate_kwargs.setdefault("output_attentions", True)

        # We constrain the generation to one blank during beam search. If we don't constrain it, it produces
        # gibberish after generating the first blank value. During beam search, the beams may differ only in
        # gibberish, that's why we want to constrain it.
        #
        # Also, `generate` now returns the logits. So we need it also for greedy search to correctly compute the
        # probabilities.
        self.generate_kwargs.setdefault("prefix_allowed_tokens_fn", self._prefix_allowed_ids)

        self.extra_id_0 = self.tokenizer.convert_tokens_to_ids(["<extra_id_0>"])[0]
        self.extra_id_1 = self.tokenizer.convert_tokens_to_ids(["<extra_id_1>"])[0]

        self.only_noun_phrases = only_noun_phrases
        if only_noun_phrases:
            # Note `num_beams` is needed, otherwise the flag doesn't make sense.
            self.generate_kwargs.setdefault("num_return_sequences", self.generate_kwargs["num_beams"])
            self.spacy_model = create_spacy_model()
        else:
            self.spacy_model = None

        self.all_token_ids = torch.arange(self.t5_pretrained_model.config.vocab_size)

    @overrides
    def on_epoch_start(self) -> None:
        self.accuracy.reset()

    @overrides
    def forward(self, masked_caption_ids: torch.Tensor, masked_caption_attention_mask: torch.Tensor,
                label_ids: Optional[torch.Tensor] = None,
                **kwargs) -> Seq2SeqLMOutput:
        return self.t5_pretrained_model(masked_caption_ids, attention_mask=masked_caption_attention_mask,
                                        labels=label_ids, **kwargs)

    def training_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> torch.Tensor:
        if batch.get('visual'):
            return self(batch["masked_caption_ids"], batch.get("masked_caption_attention_mask"),
                        batch.get("label_ids"), visual=batch.get(visual))["loss"]
        else:
            return self(batch["masked_caption_ids"], batch.get("masked_caption_attention_mask"),
                        batch.get("label_ids"))["loss"]

    def _prefix_allowed_ids(self, _batch_id: int, input_ids: torch.Tensor) -> Sequence[int]:
        return [self.tokenizer.eos_token_id] if input_ids[-1] == self.extra_id_1 else self.all_token_ids

    def _generative_step(self, masked_caption_ids: torch.Tensor, masked_caption_attention_mask: torch.Tensor,
                         label_ids: torch.Tensor, masked_caption: Sequence[str], label: Sequence[str], 
                         visual: torch.Tensor = None, **_kwargs) -> None:
        self.write_prediction("masked_caption", masked_caption)

        # Generate an answer from scratch:
        if visual is not None:
            self.generate_kwargs['visual'] = visual
        generated_output = self.t5_pretrained_model.generate(masked_caption_ids,
                                                            attention_mask=masked_caption_attention_mask,
                                                            **self.generate_kwargs)
        if visual is not None:
            generated_ids = generated_output.sequences
        #FIXME: I don't know why the output of multi-modal model is already a tensor
        else:
            generated_ids = generated_output
        generated = self.tokenizer.batch_decode(
            compute_first_blank(generated_ids, self.t5_pretrained_model.config.decoder_start_token_id,
                                self.extra_id_0, self.extra_id_1))

        generated_logits = torch.stack(generated_output.scores, dim=1)

        num_return_sequences = self.generate_kwargs.get("num_return_sequences", 1)

        if self.only_noun_phrases:
            assert len(generated) % num_return_sequences == 0

            generated_in_chunks = list(chunks(generated, num_return_sequences))

            # We use "expand" to include the punctuation or spacing even when it gets mixed with the next token.
            # So we penalize the model for generating this extra unnecessary stuff.
            noun_phrase_indices = list(arg_noun_phrase(self.spacy_model, masked_caption, generated_in_chunks,
                                                       span_alignment_mode="expand"))

            batch_size = masked_caption_ids.shape[0]
            selected_indices = torch.arange(batch_size), torch.tensor(noun_phrase_indices)

            generated_ids = generated_ids.view(batch_size, num_return_sequences, -1)[selected_indices]
            generated = [generated_instance[i]
                         for generated_instance, i in zip(generated_in_chunks, noun_phrase_indices)]

            generated_logits = generated_logits.view(batch_size, num_return_sequences, generated_logits.shape[1],
                                                     -1)[selected_indices]
        else:
            assert num_return_sequences == 1

        # We ignore the end-of-stream token as it's the end of the generation stream, not the end of the blank.
        # The end of the blank is marked by the next extra token.
        # The end of the generation stream was defined on how the model was trained.
        # It's irrelevant when computing the joint probability.
        generated_prob = compute_answer_prob(generated_logits, generated_ids, self.t5_pretrained_model.config,
                                             ignore_eos_token=True)

        self.write_prediction("generated", generated)
        self.write_prediction("generated_prob", generated_prob)

        accuracy = self.accuracy(generated, label)
        self.log("accuracy_step", accuracy, prog_bar=True)

        # Compute the ground truth likelihood:

        self.write_prediction("ground_truth", label)

        model_kwargs = {}

        if self.t5_pretrained_model.config.is_encoder_decoder:
            # Reuse the encoder output to avoid computing it twice.
            model_kwargs["encoder_outputs"] = BaseModelOutput(
                last_hidden_state=generated_output.encoder_hidden_states[-1],
                hidden_states=generated_output.encoder_hidden_states,
                attentions=generated_output.encoder_attentions,
            )

        label_ids[label_ids == self.t5_pretrained_model.config.pad_token_id] = -100  # Mask for the loss computation.
        label_output = self(masked_caption_ids, masked_caption_attention_mask, label_ids, **model_kwargs)
        self.log("loss", label_output.loss, prog_bar=True)
        label_ids[label_ids == -100] = self.t5_pretrained_model.config.pad_token_id  # Mask for the loss computation.

        label_prob = compute_answer_prob(label_output.logits, label_ids, self.t5_pretrained_model.config,
                                         ignore_eos_token=True)
        self.write_prediction("ground_truth_prob", label_prob)

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

    @overrides
    def configure_optimizers(self) -> Union[Iterable[Optimizer], Tuple[Iterable[Optimizer], Iterable[_LRScheduler]]]:
        optimizer = AdamW(self.parameters(), lr=self.optimizer_args['lr'], betas=(self.optimizer_args['beta1'], self.optimizer_args['beta2']),
                          weight_decay=self.optimizer_args['weight_decay'])
        if self.optimizer_args['lr_scheduling']:
            if self.optimizer_args['lr_scheduling'] == "linear_with_warmup":
                scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * self.optimizer_args['epochs'], self.optimizer_args['epochs'])
            else:
                raise ValueError(f"Unrecognized LR Scheduling {self.optimizer_args['lr_scheduling']}")
            return [optimizer], [scheduler]
        else:
            return [optimizer]
