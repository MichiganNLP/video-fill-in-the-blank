#!/usr/bin/env python
import random
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, AutoModelWithLMHead, get_linear_schedule_with_warmup
from transformers.modeling_auto import MODEL_FOR_PRETRAINING_MAPPING

from argparse_with_defaults import ArgumentParserWithDefaults
from data_loader_multimodal import ActivityNetCaptionDataset
from utils import batch_padding


class MultiModalLightningModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        print(hparams)

        self.transformer = AutoModelWithLMHead.from_pretrained(self.hparams.model_name)
        self.text_embedding = self.transformer.get_input_embeddings()

        embedding_size = self.text_embedding.embedding_dim
        self.video_embedding = nn.Linear(self.hparams.V_D_in, embedding_size)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)

    def forward(self, text_feature, video_feature, attention_mask, segment_mask, mask_lm_labels, position_embedding):
        video_feature_embeddings = self.video_embedding(video_feature)
        text_feature_embeddings = self.text_embedding(text_feature)

        input_feature = torch.cat([text_feature_embeddings, video_feature_embeddings], dim=1)
        out = self.transformer(inputs_embeds=input_feature, attention_mask=attention_mask, token_type_ids=segment_mask,
                               masked_lm_labels=mask_lm_labels, position_ids=position_embedding)
        return out

    def _step(self, batch):
        text_features, video_features, attention_mask, segment_mask, labels, mask_positions, mask_lm_labels, \
            position_embedding, _ = batch

        output = self.forward(text_features, video_features, attention_mask, segment_mask, mask_lm_labels,
                              position_embedding)
        loss = output[0]
        accuracy = self.__accuracy(text_features, output[1], labels, mask_positions)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
            accuracy = accuracy.unsqueeze(0)

        return accuracy, loss

    def training_step(self, batch, batch_idx):
        accuracy, loss = self._step(batch)

        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'loss': loss,
            'accuracy': accuracy,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output

    def validation_step(self, batch, batch_idx):
        accuracy, loss = self._step(batch)

        output = OrderedDict({
            'val_loss': loss,
            'val_accuracy': accuracy,
        })

        return output

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}

        for metric_name in outputs[0]:
            metric_total = 0

            for output in outputs:
                metric_value = output[metric_name]

                metric_total += metric_value

            tqdm_dict[metric_name] = metric_total / len(outputs)

        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
        return result

    def __accuracy(self, textFeatures, score, labels, mask_positions) -> torch.Tensor:
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            correct = 0
            total_num = 0
            batch_size = textFeatures.shape[0]
            predicted_index = torch.argmax(score[list(range(batch_size)), mask_positions], dim=1)

            out_text = self.tokenizer.decode(predicted_index.tolist())
            total_num += batch_size
            for i in range(batch_size):
                if labels[i] == out_text[i]:
                    correct += 1

            t = torch.empty(1, dtype=score.dtype, device=score.device)
            t.fill_(correct / total_num)

            return t

    def configure_optimizers(self):
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

    def train_dataloader(self):
        train_dataset = ActivityNetCaptionDataset(self.hparams.data_path)
        train_loader = DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                                  collate_fn=lambda b: batch_padding(b, tokenizer=self.tokenizer),
                                  num_workers=self.hparams.num_workers)
        return train_loader

    def val_dataloader(self):
        val_dataset = ActivityNetCaptionDataset(self.hparams.data_path)
        val_loader = DataLoader(val_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                                collate_fn=lambda b: batch_padding(b, tokenizer=self.tokenizer),
                                num_workers=self.hparams.num_workers)
        return val_loader

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParserWithDefaults(parents=[parent_parser])
        parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=42, help='seed for initializing training. ')
        parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N',
                            help='mini-batch size, this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--beta1', default=0.9, type=float, metavar='M1',
                            help='beta1 for adam optimizer')
        parser.add_argument('--beta2', default=0.999, type=float, metavar='M2',
                            help='beta2 for adam optimizer')
        parser.add_argument('--V-D-in', default=500, type=int, metavar='V',
                            help='input video feature dimension')
        parser.add_argument('--num_workers', default=8, type=int, help='number of workers used for data loading')
        parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                            metavar='W', help='weight decay',
                            dest='weight_decay')
        model_name_choices = sorted([key
                                     for config in MODEL_FOR_PRETRAINING_MAPPING
                                     for key in config.pretrained_config_archive_map])
        parser.add_argument('--model-name', help='transformer model to use', choices=model_name_choices,
                            default='bert-base-uncased')
        parser.add_argument('--lr-scheduling', choices=['linear_with_warmup', ''], default='linear_with_warmup')
        return parser


def get_args():
    parent_parser = ArgumentParserWithDefaults(add_help=False)
    parent_parser.add_argument('--data-path', metavar='DIR', help='path to dataset')
    parent_parser.add_argument('--save-path', metavar='DIR', default=".", help='path to save output')
    parent_parser.add_argument('--gpus', type=int, default=1, help='how many gpus')
    parent_parser.add_argument('--distributed-backend', default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--use-16bit', action='store_true', help='if true uses 16 bit precision')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('--amp-level', choices=('O0', 'O1', 'O2', 'O3'), default='O1')

    parser = MultiModalLightningModel.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(hparams):
    model = MultiModalLightningModel(hparams)
    if hparams.seed is not None:
        random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        np.random.seed(hparams.seed)
        cudnn.deterministic = True
    trainer = pl.Trainer(
        default_save_path=hparams.save_path,
        gpus=hparams.gpus,
        max_epochs=hparams.epochs,
        distributed_backend=hparams.distributed_backend,
        use_amp=hparams.use_16bit
    )
    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main(get_args())
