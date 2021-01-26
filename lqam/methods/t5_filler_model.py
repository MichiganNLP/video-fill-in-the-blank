<<<<<<< lqam/methods/t5_filler_model.py
<<<<<<< lqam/methods/t5_filler_model.py
from typing import Any, Mapping, Optional, Sequence
=======
from typing import Any, Mapping, Optional, Sequence, Union, Iterable, Tuple
>>>>>>> lqam/t5_module.py

import pytorch_lightning as pl
import torch
from overrides import overrides
<<<<<<< lqam/methods/t5_filler_model.py
from transformers import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from lqam.core.noun_phrases import create_spacy_model
from lqam.methods.dataset import TYPE_BATCH
from lqam.methods.decoding import arg_noun_phrase, compute_answer_prob
from lqam.methods.metrics import AlmostExactMatchAccuracy
from lqam.methods.t5_format_processing import compute_first_blank
from lqam.util.iterable_utils import chunks
=======
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from transformers import AdamW, get_linear_schedule_with_warmup, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import Seq2SeqLMOutput

from lqam.data_module import TYPE_BATCH
from lqam.decoder_utils import compute_label_prob
from lqam.metrics import AlmostExactMatchAccuracy
from lqam.t5_format_processing import compute_first_blank
>>>>>>> lqam/t5_module.py


# Some things were copied from https://github.com/huggingface/transformers/blob/8062fa6/examples/rag/finetune_rag.py#L94
class T5FillerModel(pl.LightningModule):
<<<<<<< lqam/methods/t5_filler_model.py
    def __init__(self, t5_like_pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
=======
    def __init__(self, t5_like_pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, optimizer_args: Mapping[str, Any],
>>>>>>> lqam/t5_module.py
                 only_noun_phrases: bool = False, generate_kwargs: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        # TODO: hparams
        # The model doesn't necessarily use T5 classes (e.g., `T5PreTrainedModel`).
        # It just needs to be pretrained like T5 and support conditional generation.
        assert isinstance(t5_like_pretrained_model, tuple(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.values()))
        self.t5_pretrained_model = t5_like_pretrained_model
        self.tokenizer = tokenizer
        self.accuracy = AlmostExactMatchAccuracy()
<<<<<<< lqam/methods/t5_filler_model.py
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

=======
        self.optimizer_args = optimizer_args
        self.generate_kwargs = generate_kwargs or {}

>>>>>>> lqam/t5_module.py
        self.extra_id_0 = self.tokenizer.convert_tokens_to_ids(["<extra_id_0>"])[0]
        self.extra_id_1 = self.tokenizer.convert_tokens_to_ids(["<extra_id_1>"])[0]

        self.only_noun_phrases = only_noun_phrases
        if only_noun_phrases:
            # Note `num_beams` is needed, otherwise the flag doesn't make sense.
            self.generate_kwargs.setdefault("num_return_sequences", self.generate_kwargs["num_beams"])
<<<<<<< lqam/methods/t5_filler_model.py
            self.spacy_model = create_spacy_model()
        else:
            self.spacy_model = None

        self.all_token_ids = torch.arange(self.t5_pretrained_model.config.vocab_size)
=======

        self.all_token_ids = torch.arange(self.tokenizer.vocab_size)
        if self.generate_kwargs.get("num_beams", 1) > 1:
            # We constrain the generation to one blank during beam search. If we don't constrain it, it produces
            # gibberish after generating the first blank value. During beam search, the beams may differ only in
            # gibberish, that's why we want to constrain it. For greedy search it doesn't change anything and it just
            # makes it slower, so we disable it.
            self.generate_kwargs.setdefault("prefix_allowed_tokens_fn", self._prefix_allowed_ids)
>>>>>>> lqam/t5_module.py

    @overrides
    def on_epoch_start(self) -> None:
        self.accuracy.reset()

    @overrides
<<<<<<< lqam/methods/t5_filler_model.py
    def forward(self, masked_caption_ids: torch.Tensor, masked_caption_attention_mask: torch.Tensor,
                label_ids: Optional[torch.Tensor] = None,
                **kwargs) -> Seq2SeqLMOutput:
        return self.t5_pretrained_model(masked_caption_ids, attention_mask=masked_caption_attention_mask,
                                        labels=label_ids, **kwargs)

    def training_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> torch.Tensor:
        return self(batch["masked_caption_ids"], batch.get("masked_caption_attention_mask"),
                    batch.get("label_ids"))["loss"]
=======
    def forward(self, masked_caption_ids: torch.Tensor, label_ids: Optional[torch.Tensor] = None,
                **kwargs) -> Seq2SeqLMOutput:
        return self.t5_pretrained_model(masked_caption_ids, labels=label_ids, **kwargs)

    def _step(self, masked_caption_ids: torch.Tensor, label_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self(masked_caption_ids, label_ids, **kwargs)["loss"]

    def training_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> torch.Tensor:
        return self._step(**batch)
>>>>>>> lqam/t5_module.py

    def _prefix_allowed_ids(self, _batch_id: int, input_ids: torch.Tensor) -> Sequence[int]:
        return [self.tokenizer.eos_token_id] if input_ids[-1] == self.extra_id_1 else self.all_token_ids

<<<<<<< lqam/methods/t5_filler_model.py
    def _generative_step(self, masked_caption_ids: torch.Tensor, masked_caption_attention_mask: torch.Tensor,
                         label_ids: torch.Tensor, masked_caption: Sequence[str], label: Sequence[str],
                         **_kwargs) -> None:
        self.write_prediction("masked_caption", masked_caption)

        # Generate an answer from scratch:

        generated_output = self.t5_pretrained_model.generate(masked_caption_ids,
                                                             attention_mask=masked_caption_attention_mask,
                                                             **self.generate_kwargs)
        generated_ids = generated_output.sequences
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
=======
    def _generative_step(self, masked_caption_ids: torch.Tensor, label_ids: torch.Tensor, masked_caption: str,
                         label: str, visual: torch.Tensor = None, **_kwargs) -> None:
        self.write_prediction("masked_caption", masked_caption)
        self.write_prediction("ground_truth", label)

        model_kwargs = {}

        if 'attention_mask' in _kwargs:
            attention_mask = _kwargs['attention_mask']
        else:
            attention_mask = None

        if self.t5_pretrained_model.config.is_encoder_decoder:
            # Compute the encoder part only once.
            # For the `generate` method it doesn't apply because it always computes it. We can't do much w/o
            # implementing our own version.
            encoder = self.t5_pretrained_model.get_encoder()
            if visual is not None:
                model_kwargs["encoder_outputs"] = encoder(masked_caption_ids, visual=visual, attention_mask=attention_mask)
            else:
                model_kwargs["encoder_outputs"] = encoder(masked_caption_ids)        

        if visual is None:
            label_output = self(masked_caption_ids, label_ids=label_ids, **model_kwargs)
        else:
            label_output = self(masked_caption_ids, label_ids=label_ids, visual=visual, attention_mask=attention_mask, **model_kwargs)
        self.log("loss", label_output["loss"], prog_bar=True)

        # We ignore the EOS token as it's about the EOS of the generation stream, not the end of the blank.
        # The end of the blank is marked by the next extra token.
        # The end of the generation stream was defined on how the model was trained.
        # It's irrelevant when computing the joint probability.
        label_prob = compute_label_prob(label_output["logits"], label_ids,
                                        pad_token_id=self.t5_pretrained_model.config.pad_token_id,
                                        eos_token_id=self.t5_pretrained_model.config.eos_token_id)
        self.write_prediction("ground_truth_prob", label_prob)

        generated_ids = self.t5_pretrained_model.generate(masked_caption_ids, visual = visual,attention_mask=attention_mask, **self.generate_kwargs)
        generated = self.tokenizer.batch_decode(
            compute_first_blank(generated_ids, self.t5_pretrained_model.config.decoder_start_token_id,
                                self.extra_id_0, self.extra_id_1))

        if self.only_noun_phrases:
            raise NotImplementedError  # TODO: filter `generated` by the first noun phrase or else the first one.

        self.write_prediction("generated", generated)

        if "prefix_allowed_tokens_fn" in self.generate_kwargs:
            # The generation was constrained, so it's already clean. But we need to remove the BoS token used to
            # start the generation. I noticed later that there's some `view(...)` in T5 code that fails due to the
            # inside  stride after doing this. So I added a `clone()`.
            clean_generated_ids = generated_ids[:, 1:].clone()
        else:
            clean_generated_ids = self.tokenizer(
                ["<extra_id_0> " + generated_instance + " <extra_id_1>" for generated_instance in generated],
                return_tensors="pt", truncation=True, padding="longest")["input_ids"].to(generated_ids.device)

        generated_output = self(masked_caption_ids, clean_generated_ids, **model_kwargs)
        generated_prob = compute_label_prob(generated_output["logits"], clean_generated_ids,
                                            pad_token_id=self.t5_pretrained_model.config.pad_token_id,
                                            eos_token_id=self.t5_pretrained_model.config.eos_token_id)
>>>>>>> lqam/t5_module.py
        self.write_prediction("generated_prob", generated_prob)

        accuracy = self.accuracy(generated, label)
        self.log("accuracy_step", accuracy, prog_bar=True)

<<<<<<< lqam/methods/t5_filler_model.py
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

=======
>>>>>>> lqam/t5_module.py
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
<<<<<<< lqam/methods/t5_filler_model.py
=======
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack
import pytorch_lightning as pl
import torch
import torch.nn as nn
from overrides import overrides
from lqam.t5_module import T5FillerModel

class NewEncoder(nn.Module):
    def __init__(self, t5_stack: T5Stack, text_embedding, visual_size: int) -> None:
        super().__init__()
        self.t5_stack = t5_stack
        self.text_embedding = text_embedding
        self.video_embedding = nn.Linear(visual_size, self.text_embedding.embedding_dim)

    def forward(self, text_token_ids, *args, **kwargs):
        text_embedding = self.text_embedding(text_token_ids)
        visual_embedding = self.video_embedding(kwargs['visual'])
        del kwargs['visual']
        embedding = torch.cat([text_embedding, visual_embedding], dim=1)
        return self.t5_stack.forward(inputs_embeds=embedding, *args, **kwargs)


class T5AndI3D(T5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        self.visual_size = kwargs['visual_size']
        del kwargs['visual_size']
        super().__init__(*args, **kwargs)        

    def set_encoder(self):
        self.encoder = NewEncoder(self.encoder, self.encoder.get_input_embeddings(), self.visual_size)
    
    @overrides
    def forward(self, masked_caption_ids=None, visual = None, labels = None, *args, **kwargs):
        if "encoder_outputs" not in kwargs:
            kwargs["encoder_outputs"] = self.encoder(masked_caption_ids, visual=visual)
        if "return_dict" not in kwargs:
            kwargs["return_dict"] = True
        return super().forward(labels = labels, *args, **kwargs)
>>>>>>> lqam/t5_multi_modal_module.py
=======
    
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
>>>>>>> lqam/t5_module.py
