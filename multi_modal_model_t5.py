import argparse
from transformers import AdamW, T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from typing import Any, Dict, Iterable, Mapping, Tuple, TypeVar, Union, MutableMapping, Optional, Sequence
import logging
import os
import pickle
import re
import random
import numpy as np
from overrides import overrides
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from argparse_with_defaults import ArgumentParserWithDefaults
from VATEX_dataset import VATEX_Dataset
import torch
from torch.optim import Optimizer
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

FRAMEWORK = "pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_NUMBER = 0 if DEVICE == "cuda" else -1

T = TypeVar("T")
TYPE_BATCH = Sequence[Tuple[Any, Any, Any, Any]]
TYPE_STEP_OUTPUT = MutableMapping[str, torch.Tensor]

logger = logging.getLogger(__name__)

class VATEXLightningModel(LightningModule):
    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.hparams = hparams
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.transformer_model_name)

        self.encoder = T5ForConditionalGeneration.from_pretrained(hparams.transformer_model_name)
        self.text_embedding = self.encoder.get_input_embeddings()
        self.video_embedding = nn.Linear(self.hparams.visual_size, self.text_embedding.embedding_dim)
        self.RE_EXTRA_ID = re.compile(r"<extra_id_\d+>")

    # From https://stackoverflow.com/a/5434936/1165181
    def pairwise(self, iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)
    
    def compute_mask_values(self, generated_ids: torch.Tensor, tokenizer: T5Tokenizer) -> Mapping[str, Iterable[str]]:
        tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
        extra_id_indices = {token: i for i, token in enumerate(tokens) if self.RE_EXTRA_ID.match(token)}
        extra_id_indices["</s>"] = len(tokens)

        return {extra_id_token: tokens[extra_id_indices[extra_id_token] + 1:extra_id_indices[next_extra_id_token]]
                for extra_id_token, next_extra_id_token in self.pairwise(extra_id_indices)}

    def match(self, pred: str, label: str) -> bool:
        return pred.lower() == label.lower()

    @overrides
    def forward(self, text_token_ids: torch.Tensor, visual: Optional[torch.Tensor], attention_mask: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        text_embedding = self.text_embedding(text_token_ids)
        
        
        if visual is None:
            embedding = text_embedding
        else:
            visual_embedding = self.video_embedding(visual)
            embedding = torch.cat([text_embedding, visual_embedding], dim=1)
        ## for debug
        generated_ids = self.greedy_search(embedding, attention_mask)
        
        if self.training:
            return self.encoder(inputs_embeds=embedding, attention_mask=attention_mask, labels = labels)
        else:
            generated_ids = self.encoder.generate(inputs_embeds=embedding, attention_mask=attention_mask)
            return compute_mask_values(generated_ids, self.tokenizer)['<extra_id_0>']

    def _train_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_token_ids, video_features, attention_masks, labels = batch
        loss, logits = self.forward(text_token_ids, video_features, attention_masks, labels)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        return loss

    def _testval_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_token_ids, video_features, attention_masks, labels = batch
        batch_size = text_token_ids.shape[0]

        correct = 0
        for i in range(batch_size):
            pred_tokens = self.forward(text_token_ids, video_features, attention_masks, labels)
            if self.match(pred_tokens):
                correct += 1

        torch.tensor(correct, dtype=torch.int64, device=text_token_ids.device)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
            correct = correct.unsqueeze(0)

        batch_size = torch.empty_like(correct)
        batch_size.fill_(scores.shape[0])

        dtype = loss.dtype
        accuracy = correct.to(dtype=dtype) / batch_size.to(dtype=dtype)

        return accuracy, correct, batch_size


    @overrides
    def configure_optimizers(self) -> Union[Iterable[Optimizer], Tuple[Iterable[Optimizer], Iterable[_LRScheduler]]]:
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2),
                          weight_decay=self.hparams.weight_decay)
        if self.hparams.lr_scheduling:
            if self.hparams.lr_scheduling == "linear_with_warmup":
                scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * self.hparams.epochs, self.hparams.epochs)
            else:
                raise ValueError(f"Unrecognized LR Scheduling {self.hparams.lr_scheduling}")
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    @overrides
    def training_step(self, batch: TYPE_BATCH,
                      batch_idx: int) -> Union[int, MutableMapping[str, Union[torch.Tensor, TYPE_STEP_OUTPUT]]]:
        accuracy, correct, batch_size, loss = self._train_step(batch)
        metrics_to_show_and_log = {"train_loss": loss}
        return {
            "loss": loss,
            "progress_bar": metrics_to_show_and_log,
            "log": metrics_to_show_and_log,
        }

    @overrides
    def validation_step(self, batch: Tuple[Any], batch_idx: int) -> TYPE_STEP_OUTPUT:
        accuracy, correct, batch_size = self._testval_step(batch)
        return {"val_accuracy": accuracy, "correct": correct, "batch_size": batch_size}

    @overrides
    def test_step(self, batch: Tuple[Any], batch_idx: int) -> TYPE_STEP_OUTPUT:
        accuracy, correct, batch_size, loss = self._testval_step(batch)
        return {"test_accuracy": accuracy, "correct": correct, "batch_size": batch_size}


    def _average_metrics(self, step_outputs: Sequence[TYPE_STEP_OUTPUT], key_prefix: str = "") -> TYPE_STEP_OUTPUT:
        loss_key = f"{key_prefix}loss"
        metrics: TYPE_STEP_OUTPUT = {}

        if  key_prefix == 'train_':
            metric_names = {loss_key}
        else:
            metric_names = {"correct", "batch_size"}
        
        for metric_name in metric_names:
            metric_total = 0

            for output in step_outputs:
                metric_value = output[metric_name]

                # reduce manually when using dp
                if self.trainer.use_dp or self.trainer.use_ddp2:
                    if metric_name in {"correct", "batch_size"}:
                        metric_value = metric_value.sum()
                    else:
                        metric_value = metric_value.mean()

                metric_total += metric_value

            if metric_name == loss_key:
                metrics[metric_name] = metric_total / len(step_outputs)  # noqa
            else:
                metrics[metric_name] = metric_total  # noqa

        dtype = metrics[loss_key].dtype

        if key_prefix != 'train_':
            metrics[f"{key_prefix}acc"] = metrics["correct"].to(dtype=dtype) / metrics["batch_size"].to(dtype=dtype)

            del metrics["correct"]
            del metrics["batch_size"]

        return metrics

    @overrides
    def training_epoch_end(self, outputs: Sequence[TYPE_STEP_OUTPUT]) -> MutableMapping[str, TYPE_STEP_OUTPUT]:
        metrics = self._average_metrics(outputs, "train_")
        return {"progress_bar": metrics, "log": metrics}

    @overrides
    def validation_epoch_end(self, outputs: Sequence[TYPE_STEP_OUTPUT]) -> MutableMapping[str, TYPE_STEP_OUTPUT]:
        metrics = self._average_metrics(outputs, "val_")
        return {"progress_bar": metrics, "log": metrics}

    @overrides
    def test_epoch_end(self, outputs: Sequence[TYPE_STEP_OUTPUT]) -> MutableMapping[str, TYPE_STEP_OUTPUT]:
        metrics = self._average_metrics(outputs, "test_")
        return {"progress_bar": metrics, "log": metrics}

    def _pad_batch(self, batch: Sequence[Sequence[Any]]) -> TYPE_BATCH:
        batch_size = len(batch)
        text_features = []
        video_features = []
        labels = []

        max_video_len = 0

        for i in range(batch_size):
            data = batch[i]
            text_features.append(data[0])
            video_features.append(data[1])
            labels.append(data[2])

            if data[1] != None:
                total_video_len = data[1].shape[0]
            else:
                total_video_len = 0
            if total_video_len > max_video_len:
                max_video_len = total_video_len
        
        text_batch = self.tokenizer.prepare_seq2seq_batch(src_texts=text_features,tgt_texts=labels, padding=True, return_tensors="pt")
        text_tensor = text_batch.input_ids
        text_attention_mask = text_batch.attention_mask
        labels = text_batch.labels

        if self.hparams.enable_visual_features:
            video_tensor = torch.zeros(batch_size, max_video_len, self.hparams.visual_size, dtype=torch.float)
        else:
            video_tensor = None

        if self.hparams.enable_visual_features:
            video_attention_mask = torch.zeros(batch_size, max_video_len, dtype=torch.long)

        for i in range(batch_size):
            video = video_features[i]
            video_len = len(video)

            # The input to the transformer is gonna be:
            # t_1 ... t_n pad ... pad </s> v_1 ... v_m pad ... pad

            if self.hparams.enable_visual_features and video != None:
                video_len = video.shape[0]
                video_tensor[i, :video_len] = video
                video_attention_mask[i, :video_len] = True

        attention_mask = torch.cat([text_attention_mask, video_attention_mask], 1)

        return (text_tensor, video_tensor, attention_mask, labels)


    def _dataloader(self, pickle_path_inside_data_folder: str) -> DataLoader:

        path = os.path.join(self.hparams.data_path, pickle_path_inside_data_folder)
        dataset = VATEX_Dataset(path)

        if self.training:
            shuffle = self.hparams.overfit_pct == 0
        else:
            shuffle = False

        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=shuffle, collate_fn=self._pad_batch,
                          pin_memory=self.hparams.pin_memory, num_workers=self.hparams.num_workers)

    @overrides
    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train.pkl")

    @overrides
    def val_dataloader(self) -> DataLoader:
        return self._dataloader("val.pkl")

    @overrides
    def test_dataloader(self) -> DataLoader:
        return self._dataloader("test.pkl")

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = ArgumentParserWithDefaults(parents=[parent_parser], add_help=False)
        parser.add_argument("--disable-visual-features", dest="enable_visual_features", action="store_false")
        parser.add_argument("--overfit-pct", default=0, type=float, help="How much of the data loaders to use. "
                                                                         "Useful for debugging. Data shuffling is "
                                                                         "disabled if it's non-zero")
        parser.add_argument("--num-workers", default=0, type=int, help="number of workers used for data loading")
        parser.add_argument("--pin-memory", action="store_true")
        parser.add_argument("--batch-size", default=16, type=int, metavar="N",
                            help="mini-batch size. This is the total batch size of all GPUs on the current node when "
                                 "using Data Parallel or Distributed Data Parallel")
        parser.add_argument("--beta1", default=0.9, type=float, help="beta1 for the Adam optimizer")
        parser.add_argument("--beta2", default=0.999, type=float, help="beta2 for the Adam optimizer")
        parser.add_argument("--epochs", default=10, type=int)
        parser.add_argument("--lr", default=0.0001, type=float)
        parser.add_argument("--lr-scheduling", choices=("", "linear_with_warmup"), default="linear_with_warmup")
        parser.add_argument("--seed", type=int, default=42)
        model_name_choices = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]
        parser.add_argument("--transformer-model-name", choices=model_name_choices, default="t5-base")
        parser.add_argument("--visual-size", default=1024, type=int, metavar="V", help="input video feature dimension")
        parser.add_argument("--weight-decay", default=1e-4, type=float)
        parser.add_argument("--max-length", default=512, type=int, help="maxium input length")
        return parser

    def greedy_search(
        self,
        input_embeds: torch.LongTensor, # N * T * V
        attention_mask: torch.LongTensor,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):

        # init values
        max_length = max_length if max_length is not None else self.hparams.max_length
        #pad_token_id = pad_token_id if pad_token_id is not None else self.tokenizer.convert_tokens_to_ids("<pad>")
        pad_token_id = 0 # for convenience, set <pad> to zero (which is default as well)
        eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.convert_tokens_to_ids("</s>")

        cur_len = input_embeds.shape[1]

        generated_ids = None
        pad_mask = attention_mask.new(attention_mask.shape[0], 1).fill_(1)
        eos_mask = attention_mask.new(attention_mask.shape[0], 1).fill_(1)
        decoder_inputs = torch.tensor(input_embeds.shape[0], 1).fill_(pad_token_id).to(DEVICE).long()

        while cur_len < max_length:

            # forward pass to get next token
            outputs = self.encoder(inputs_embeds=input_embeds, attention_mask=attention_mask, decoder_input_ids=decoder_inputs, return_dict=True)
            pad_mask = torch.mul(pad_mask, eos_mask)
            attention_mask = torch.cat([attention_mask, pad_mask], dim = 1)
            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            scores = F.softmax(next_token_logits)

            # argmax
            next_tokens = torch.argmax(scores, dim=-1)

            # concat generated text to decoder inputs
            decoder_inputs = torch.cat([decoder_inputs, next_tokens], dim=1)
            
            # save generated ids
            if generated_ids is None:
                generated_ids = next_tokens
            else:
                torch.cat([generated_ids, next_tokens], dim=1)

            # prepare to mask sentences that reaches </s>
            eos_mask = torch.mul(eos_mask, (next_tokens != eos_token_id))

            if eos_mask.sum() == 0:
                break

            # increase cur_len
            cur_len = cur_len + 1

        return generated_ids

def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True

class SortingHelpFormatter(argparse.HelpFormatter):
    @overrides
    def add_arguments(self, actions: Iterable[argparse.Action]) -> None:
        sorted_actions = sorted(actions, key=lambda a: a.option_strings)
        super().add_arguments(sorted_actions)

def _get_args() -> argparse.Namespace:
    parent_parser = ArgumentParserWithDefaults(formatter_class=SortingHelpFormatter)  # noqa
    parent_parser.add_argument("--amp-level", choices=("O0", "O1", "O2", "O3"), default="O1",
                               help="only on when --use-16bit is on")
    parent_parser.add_argument("--data-path", metavar="DIR", default="data")
    parent_parser.add_argument("--distributed-backend", default="dp", choices=("dp", "ddp", "ddp2"))
    parent_parser.add_argument("--fast-dev-run", action="store_true")
    parent_parser.add_argument("--gpu-count", type=int, default=1, help="gpu count")
    parent_parser.add_argument("--resume-from-checkpoint", metavar="CHECKPOINT_FILE")
    parent_parser.add_argument("--save-path", metavar="DIR", default=".")
    parent_parser.add_argument("--use-16bit", action="store_true")
    parent_parser.add_argument("-v", "--verbose", action="store_true")
    parser = VATEXLightningModel.add_model_specific_args(parent_parser)
    return parser.parse_args()

def _main() -> None:
    hparams = _get_args()

    if hparams.seed is not None:
        _set_seed(hparams.seed)

    logging_level = logging.INFO if hparams.verbose else logging.WARNING
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging_level)

    logger.info(hparams)

    model = VATEXLightningModel(hparams)
    trainer = pl.Trainer(default_save_path=hparams.save_path, gpus=hparams.gpu_count, max_epochs=hparams.epochs,
                        distributed_backend=hparams.distributed_backend, use_amp=hparams.use_16bit, benchmark=True,
                        amp_level=hparams.amp_level, resume_from_checkpoint=hparams.resume_from_checkpoint,
                        progress_bar_refresh_rate=1, overfit_pct=hparams.overfit_pct,
                        fast_dev_run=hparams.fast_dev_run)
    trainer.fit(model)
    trainer.test(model)
                


if __name__ == "__main__":
    _main()