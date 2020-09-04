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

TYPE_BATCH = Sequence[Tuple[Any, Any, Any, Any, Any, Any, Any, Any]]
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
        # text_token_ids shape: (batch_size, max_seq_len + 1)
        # mask shape: (batch_size, max_seq_len + 1)
        # mask_lm_labels (batch_size, max_seq_len + 1)
        # position_ids: (batch_size, max_seq_len + 1)
        raise NotImplementedError

    def _correct_predictions(self, scores: torch.Tensor, labels: Sequence[str],
                             mask_positions: torch.Tensor) -> torch.Tensor:
        # scores shape: (batch_size, max_seq_length, vocab_size)
        with torch.no_grad():
            batch_size = scores.shape[0]
            # prediction_indices = scores[list(range(batch_size)), mask_positions].argmax(dim=1)

            # predictions = self.tokenizer.convert_ids_to_tokens(prediction_indices.tolist())
            # correct = sum(prediction == label for prediction, label in zip(predictions, labels))

            # For now, I also predict a number of word pieces in during test time
            correct = 0
            for i in range(batch_size):
                predicted_indices = scores[i, mask_positions[i]:mask_positions[i] + len(labels[i])].argmax(dim=1)
                prediction = self.tokenizer.convert_ids_to_tokens(predicted_indices)
                correct += int(prediction == labels[i])

            return torch.tensor(correct, dtype=torch.int64, device=scores.device)

    def _step(self, text_token_ids: torch.Tensor, visual: Optional[torch.Tensor], mask: torch.Tensor,
              segment_mask: torch.Tensor, labels: Sequence[str], mask_positions: Sequence[int],
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

    def _val_test_step(self, batch: TYPE_BATCH):
        # scores = []
        # correct = torch.tensor(0, dtype=torch.int64, device=batch[0][0].device)
        # batch_size = batch[0][0].shape[0]

        # with torch.no_grad():
        #     for i in range(len(batch)):
        #         text_token_ids, visual, mask, segment_mask, labels, mask_positions, mask_lm_labels, position_ids = \
        #             batch[i]
        #         _, score = self.forward(text_token_ids, visual, mask, segment_mask, mask_lm_labels, position_ids)
        #         scores.append(score)

        #     # for each element in one batch
        #     for i in range(batch_size):
        #         confidence = max(scores[0][i][batch[0][5][i]])
        #         token_num = 0
        #         # for each different len
        #         for j in range(1, len(scores)):
        #             _confid = 0
        #             mask_position = batch[j][5][i]
        #             label = batch[j][4][i]
        #             for k in range(j + 1):
        #                 _confid += max(scores[j][i][mask_position + k])
        #             _confid = _confid / (j + 1)
        #             if _confid > confidence:
        #                 confidence = _confid
        #                 token_num = j

        #         # compute correctness
        #         mask_position = batch[token_num][5][i]
        #         label = batch[token_num][4][i]
        #         label_len = len(label)
        #         if label_len == token_num + 1:
        #             all_correct = True
        #             for j in range(label_len):
        #                 prediction_index = torch.argmax(scores[token_num][i, mask_position + j])
        #                 prediction = self.tokenizer.convert_ids_to_tokens(prediction_index.tolist())
        #                 if prediction != labels[i][j]:
        #                     all_correct = False
        #                     break
        #             if all_correct:
        #                 correct += 1

        #     if self.trainer.use_dp or self.trainer.use_ddp2:
        #         correct = correct.unsqueeze(0)

        #     batch_size = torch.empty_like(correct)
        #     batch_size.fill_(scores[0].shape[0])

        #     accuracy = correct / batch_size
        batch_size = batch[0][0].shape[0]
        text_token_ids, visual, mask, segment_mask, labels, mask_positions, mask_lm_labels, position_ids = batch[0]
        loss, scores = self.forward(text_token_ids, visual, mask, segment_mask, mask_lm_labels, position_ids)
        prediction_indices = torch.argmax(scores[list(range(batch_size)), mask_positions], dim=1)

        predictions = self.tokenizer.convert_ids_to_tokens(prediction_indices.tolist())
        correct = sum((len(label)==1 and prediction == label[0]) for prediction, label in zip(predictions, labels))

        correct = torch.tensor(correct, dtype=torch.int64, device=scores.device)
        if self.trainer.use_dp or self.trainer.use_ddp2:
            correct = correct.unsqueeze(0)

        batch_size = torch.empty_like(correct)
        batch_size.fill_(scores.shape[0])        

        dtype = loss.dtype
        accuracy = correct.to(dtype=dtype) / batch_size.to(dtype=dtype)

        return accuracy, correct, batch_size, loss


    @overrides
    def training_step(self, batch: TYPE_BATCH,
                      batch_idx: int) -> Union[int, MutableMapping[str, Union[torch.Tensor, TYPE_STEP_OUTPUT]]]:
        accuracy, correct, batch_size, loss = self._step(*batch[0])
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
        accuracy, correct, batch_size, loss = self._val_test_step(batch)
        return {"val_accuracy": accuracy, "correct": correct, "batch_size": batch_size, "val_loss": loss}

    @overrides
    def test_step(self, batch: Tuple[Any], batch_idx: int) -> TYPE_STEP_OUTPUT:
        accuracy, correct, batch_size, loss = self._val_test_step(*batch)
        return {"test_accuracy": accuracy, "correct": correct, "batch_size": batch_size, "test_loss": loss}


    def _average_metrics(self, step_outputs: Sequence[TYPE_STEP_OUTPUT], key_prefix: str = "") -> TYPE_STEP_OUTPUT:
        loss_key = f"{key_prefix}loss"
        metrics: TYPE_STEP_OUTPUT = {}

        metric_names = {"correct", "batch_size", loss_key}
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

    def _pad_batch(self, batch: Sequence[Sequence[Any]]) -> TYPE_BATCH:
        batch_size = len(batch)
        out = []
        max_token_count = 1 if self.training else self.hparams.max_token_num
        for token_count in range(max_token_count):
            text_features = []
            video_features = []
            labels = []
            mask_positions = []

            max_text_len = 0
            max_video_len = 0
            video = None

            for i in range(batch_size):
                data = batch[i]
                # text = torch.tensor(data[0])
                text = torch.tensor(self.tokenizer.encode(' '.join(data[0])))
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

            if not self.training:
                max_text_len += token_count

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
                if self.training:
                    text_tensor[i, :text_len - 1] = text[:-1]
                else:
                    text_tensor[i, :mask_positions[i] + 1] = text[:mask_positions[i] + 1]
                    for k in range(token_count):
                        text_tensor[i, :mask_positions[i] + 1 + k] = text[mask_positions[i]]
                    text_tensor[i, mask_positions[i] + 1 + token_count: text_len + token_count - 1] = text[mask_positions[i] + 1: -1]
                text_tensor[i, -1] = text[-1]
                if self.training:
                    mask[i, :text_len - 1] = True
                else:
                    mask[i, :text_len + token_count - 1] = True

                if self.hparams.enable_visual_features:
                    video_len = video.shape[0]
                    video_tensor[i, :video_len] = video
                    mask[i, max_text_len - 1:max_text_len + video_len] = True

                # We know label length in training. For val and testing, mask_lm_labels is not used
                if self.training:
                    label_len = len(labels[i])
                    masked_lm_labels[i, mask_positions[i]:mask_positions[i] + label_len] = torch.LongTensor(
                        self.tokenizer.convert_tokens_to_ids(labels[i]))
                else:
                    masked_lm_labels[i, mask_positions[i]] = self.tokenizer.convert_tokens_to_ids(labels[i][0])

            out.append((text_tensor, video_tensor, mask, segments_tensor, labels, mask_positions, masked_lm_labels,
                        position_ids))
        return out

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
        return self._dataloader("val.pkl")

    @overrides
    def test_dataloader(self) -> DataLoader:
        return self._dataloader("test.pkl")

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
