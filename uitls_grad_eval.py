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

from transformers import AutoTokenizer


TYPE_BATCH = Sequence[Tuple[Any, Any, Any, Any, Any, Any, Any, Any]]
TYPE_STEP_OUTPUT = MutableMapping[str, torch.Tensor]

def _pad_batch(batch: Sequence[Sequence[Any]]) -> TYPE_BATCH:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


    batch_size = len(batch)
    out = []
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
    video_tensor = torch.zeros(batch_size, max_video_len, video.shape[1], dtype=torch.float, requires_grad=True)


    segments_tensor = torch.cat([torch.zeros(batch_size, max_text_len, dtype=torch.long),
                                    torch.ones(batch_size, max_video_len, dtype=torch.long)], dim=1)
    mask = torch.zeros(batch_size, max_text_len + max_video_len, dtype=torch.bool)
    # `-100` is the default `ignore_index` value for CrossEntropyLoss.
    masked_lm_labels = torch.ones(batch_size, max_text_len + max_video_len, dtype=torch.long) * -100
    position_ids = torch.cat([torch.arange(max_text_len, dtype=torch.long),
                                torch.arange(max_video_len, dtype=torch.long)],
                                dim=0).unsqueeze(0).repeat(batch_size, 1)

    for i in range(batch_size):
        text = text_features[i]
        video = video_features[i]
        text_len = len(text)

        # The input to the transformer is gonna be:
        # [CLS] t_1 ... t_n pad ... pad [SEP] v_1 ... v_m pad ... pad [SEP]
        text_tensor[i, :text_len - 1] = text[:-1]
        text_tensor[i, -1] = text[-1]
        mask[i, :text_len - 1] = True

        video_len = video.shape[0]
        video_tensor[i, :video_len] = video
        mask[i, max_text_len - 1:max_text_len + video_len] = True

        # We know label length in training. For val and testing, mask_lm_labels is not used
        masked_lm_labels[i, mask_positions[i]] = tokenizer.convert_tokens_to_ids(labels[i][0])

    out.append((text_tensor, video_tensor, mask, segments_tensor, labels, mask_positions, masked_lm_labels,
                position_ids))
    return out

def _dataloader(pickle_path_inside_data_folder: str, hparams: argparse.Namespace) -> DataLoader:
    path = os.path.join(hparams.data_path, pickle_path_inside_data_folder)
    dataset = ActivityNetCaptionsDataset(path)

    shuffle = hparams.overfit_pct == 0

    return DataLoader(dataset, batch_size=hparams.batch_size, shuffle=shuffle, collate_fn=_pad_batch,
                        pin_memory=hparams.pin_memory, num_workers=hparams.num_workers)