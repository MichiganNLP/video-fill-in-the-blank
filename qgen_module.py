#!/usr/bin/env python
import argparse
import os
from typing import Any, MutableMapping, Optional, Sequence, Tuple, Union

import torch
from overrides import overrides
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from argparse_with_defaults import ArgumentParserWithDefaults
from data_loader_multimodal import ActivityNetCaptionsDataset

TYPE_STEP_OUTPUT = MutableMapping[str, torch.Tensor]


class QGenLightningModel(LightningModule):
    def __init__(self, tokenizer: PreTrainedTokenizer, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.hparams = hparams

    @overrides
    def forward(self, text_token_ids: torch.Tensor, visual: Optional[torch.Tensor], mask: torch.Tensor,
                segment_mask: torch.Tensor, mask_lm_labels: torch.Tensor,
                position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _correct_predictions(self, scores: torch.Tensor, labels: Sequence[str],
                             mask_positions: torch.Tensor) -> torch.Tensor:
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            batch_size = scores.shape[0]
            prediction_indices = torch.argmax(scores[list(range(batch_size)), mask_positions], dim=1)

            predictions = self.tokenizer.convert_ids_to_tokens(prediction_indices.tolist())
            correct = sum(prediction == label for prediction, label in zip(predictions, labels))

            return torch.tensor(correct, dtype=torch.int64, device=scores.device)

    def _step(self, text_token_ids: torch.Tensor, visual: Optional[torch.Tensor], mask: torch.Tensor,
              segment_mask: torch.Tensor, labels: Sequence[str], mask_positions: torch.Tensor,
              mask_lm_labels: torch.Tensor,
              position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss, scores = self.forward(text_token_ids, visual, mask, segment_mask, mask_lm_labels, position_ids)
        correct = self._correct_predictions(scores, labels, mask_positions)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
            correct = correct.unsqueeze(0)

        batch_size = torch.empty_like(correct)
        batch_size.fill_(scores.shape[0])

        dtype = loss.dtype
        accuracy = correct.to(dtype=dtype) / batch_size.to(dtype=dtype)

        return accuracy, correct, batch_size, loss

    @overrides
    def training_step(self, batch: Tuple[Any],
                      batch_idx: int) -> Union[int, MutableMapping[str, Union[torch.Tensor, TYPE_STEP_OUTPUT]]]:
        accuracy, correct, batch_size, loss = self._step(*batch)
        metrics_to_show_and_log = {"train_loss": loss, "acc": accuracy}
        return {
            "accuracy": accuracy,
            "correct": correct,
            "batch_size": batch_size,
            "loss": loss,
            "progress_bar": metrics_to_show_and_log,
            "log": metrics_to_show_and_log,
        }

    @overrides
    def validation_step(self, batch: Tuple[Any], batch_idx: int) -> TYPE_STEP_OUTPUT:
        accuracy, correct, batch_size, loss = self._step(*batch)
        return {"val_accuracy": accuracy, "correct": correct, "batch_size": batch_size, "val_loss": loss}

    @overrides
    def test_step(self, batch: Tuple[Any], batch_idx: int) -> TYPE_STEP_OUTPUT:
        accuracy, correct, batch_size, loss = self._step(*batch)
        return {"test_accuracy": accuracy, "correct": correct, "batch_size": batch_size, "test_loss": loss}

    def _average_metrics(self, step_outputs: Sequence[TYPE_STEP_OUTPUT], key_prefix: str = "") -> TYPE_STEP_OUTPUT:
        loss_key = f"{key_prefix}loss"
        metrics: TYPE_STEP_OUTPUT = {}
        for metric_name in {"correct", "batch_size", loss_key}:
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
        metrics[f"{key_prefix}acc"] = metrics["correct"].to(dtype=dtype) / metrics["batch_size"].to(dtype=dtype)

        del metrics["correct"]
        del metrics["batch_size"]

        return metrics

    @overrides
    def training_epoch_end(self, outputs: Sequence[TYPE_STEP_OUTPUT]) -> MutableMapping[str, TYPE_STEP_OUTPUT]:
        metrics = self._average_metrics(outputs)
        return {"progress_bar": metrics, "log": metrics}

    @overrides
    def validation_epoch_end(self, outputs: Sequence[TYPE_STEP_OUTPUT]) -> MutableMapping[str, TYPE_STEP_OUTPUT]:
        metrics = self._average_metrics(outputs, "val_")
        return {"progress_bar": metrics, "log": metrics}

    @overrides
    def test_epoch_end(self, outputs: Sequence[TYPE_STEP_OUTPUT]) -> MutableMapping[str, TYPE_STEP_OUTPUT]:
        metrics = self._average_metrics(outputs, "test_")
        return {"progress_bar": metrics, "log": metrics}

    def _pad_batch(self, batch: Sequence[Sequence[Any]]) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        batch_size = len(batch)

        text_features = []
        video_features = []
        labels = []
        mask_positions = []

        max_text_len = 0
        max_video_len = 0
        video = None
        for i in range(batch_size):
            data = batch[i]
            text = torch.tensor(data[0])
            video = data[1]
            labels.append(data[2])
            mask_positions.append(data[3])

            text_features.append(text)
            video_features.append(video)

            total_text_len = len(text)
            if total_text_len > max_text_len:
                max_text_len = total_text_len

            total_video_len = video.shape[0]
            if total_video_len > max_video_len:
                max_video_len = total_video_len

        text_tensor = torch.zeros(batch_size, max_text_len, dtype=torch.long)
        if self.hparams.enable_visual_features:
            video_tensor = torch.zeros(batch_size, max_video_len, video.shape[1], dtype=torch.float)
        else:
            video_tensor = None

        if self.hparams.enable_visual_features:
            segments_tensor = torch.cat([torch.zeros(batch_size, max_text_len, dtype=torch.long),
                                         torch.ones(batch_size, max_video_len, dtype=torch.long)], dim=1)
            mask = torch.zeros(batch_size, max_text_len + max_video_len, dtype=torch.bool)
            # `-100` is the default `ignore_index` value for CrossEntropyLoss.
            masked_lm_labels = torch.ones(batch_size, max_text_len + max_video_len, dtype=torch.long) * -100
            position_ids = torch.cat([torch.arange(max_text_len, dtype=torch.long),
                                      torch.arange(max_video_len, dtype=torch.long)],
                                     dim=0).unsqueeze(0).repeat(batch_size, 1)
        else:
            segments_tensor = torch.zeros(batch_size, max_text_len, dtype=torch.long)
            mask = torch.zeros(batch_size, max_text_len, dtype=torch.bool)
            # `-100` is the default `ignore_index` value for CrossEntropyLoss.
            masked_lm_labels = torch.ones(batch_size, max_text_len, dtype=torch.long) * -100
            position_ids = torch.arange(max_text_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

        for i in range(batch_size):
            text = text_features[i]
            video = video_features[i]
            text_len = len(text)

            # The input to the transformer is gonna be:
            # [CLS] t_1 ... t_n pad ... pad [SEP] v_1 ... v_m pad ... pad [SEP]

            text_tensor[i, :text_len - 1] = text[:-1]
            text_tensor[i, -1] = text[-1]
            mask[i, :text_len - 1] = True

            if self.hparams.enable_visual_features:
                video_len = video.shape[0]
                video_tensor[i, :video_len] = video
                mask[i, max_text_len - 1:max_text_len + video_len] = True

            masked_lm_labels[i, mask_positions[i]] = self.tokenizer.convert_tokens_to_ids(labels[i])

        return text_tensor, video_tensor, mask, segments_tensor, labels, mask_positions, masked_lm_labels, position_ids

    def _dataloader(self, pickle_path_inside_data_folder: str) -> DataLoader:
        path = os.path.join(self.hparams.data_path, pickle_path_inside_data_folder)
        dataset = ActivityNetCaptionsDataset(path)

        shuffle = self.hparams.overfit_pct == 0

        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=shuffle, collate_fn=self._pad_batch,
                          pin_memory=self.hparams.pin_memory, num_workers=self.hparams.num_workers)

    @overrides
    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train.pkl")

    @overrides
    def val_dataloader(self) -> DataLoader:
        return self._dataloader("val1.pkl")

    @overrides
    def test_dataloader(self) -> DataLoader:
        return self._dataloader("val2.pkl")

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:  # pragma: no-cover
        parser = ArgumentParserWithDefaults(parents=[parent_parser], add_help=False)
        parser.add_argument("--disable-visual-features", dest="enable_visual_features", action="store_false")
        parser.add_argument("--overfit-pct", default=0, type=float, help="How much of the data loaders to use. "
                                                                         "Useful for debugging. Data shuffling is "
                                                                         "disabled if it's non-zero")
        parser.add_argument("--num-workers", default=0, type=int, help="number of workers used for data loading")
        parser.add_argument("--pin-memory", action="store_true")
        return parser
