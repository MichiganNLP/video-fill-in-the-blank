#!/usr/bin/env python
import argparse
import random
from typing import Iterable, Optional, Tuple, Union

import logging
import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from overrides import overrides
from torch.optim import Optimizer  # noqa
from torch.optim.lr_scheduler import _LRScheduler  # noqa
from transformers import AdamW, AutoTokenizer, AutoModelWithLMHead, get_linear_schedule_with_warmup
from transformers.modeling_auto import MODEL_FOR_PRETRAINING_MAPPING

from argparse_with_defaults import ArgumentParserWithDefaults
from qgen_module import QGenLightningModel

from utils_grad_eval import _dataloader

from torch.autograd import Variable

logger = logging.getLogger(__name__)


class MultiModalLightningModel(QGenLightningModel):
    def __init__(self, hparams: argparse.Namespace) -> None:
        tokenizer = AutoTokenizer.from_pretrained(hparams.transformer_model_name)
        super().__init__(tokenizer=tokenizer, hparams=hparams)

        self.encoder = AutoModelWithLMHead.from_pretrained(self.hparams.transformer_model_name)
        self.text_embedding = self.encoder.get_input_embeddings()
        self.video_embedding = nn.Linear(self.hparams.visual_size, self.text_embedding.embedding_dim)

    @overrides
    def forward(self, text_token_ids: torch.Tensor, visual: Optional[torch.Tensor], mask: torch.Tensor,
                segment_mask: torch.Tensor, mask_lm_labels: torch.Tensor,
                position_ids: torch.Tensor, grad_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        text_embedding = self.text_embedding(text_token_ids)
        if visual is None:
            embedding = text_embedding
        else:
            visual_embedding = self.video_embedding(visual)
            embedding = torch.cat([text_embedding, visual_embedding], dim=1)

        # if self.grad_eval:
        #     return self.encoder(inputs_embeds=embedding, attention_mask=mask, token_type_ids=segment_mask,
        #                     masked_lm_labels=mask_lm_labels, position_ids=position_ids), text_embedding
        if grad_eval:
            embedding = Variable(embedding, requires_grad = True)
            return self.encoder(inputs_embeds=embedding, attention_mask=mask, token_type_ids=segment_mask,
                            masked_lm_labels=mask_lm_labels, position_ids=position_ids), embedding
        else:
            return self.encoder(inputs_embeds=embedding, attention_mask=mask, token_type_ids=segment_mask,
                            masked_lm_labels=mask_lm_labels, position_ids=position_ids)

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

    @classmethod
    @overrides
    def add_model_specific_args(cls, parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser2 = super().add_model_specific_args(parent_parser)

        parser = ArgumentParserWithDefaults(parents=[parent_parser2], add_help=False)
        parser.add_argument("--batch-size", default=16, type=int, metavar="N",
                            help="mini-batch size. This is the total batch size of all GPUs on the current node when "
                                 "using Data Parallel or Distributed Data Parallel")
        parser.add_argument("--beta1", default=0.9, type=float, help="beta1 for the Adam optimizer")
        parser.add_argument("--beta2", default=0.999, type=float, help="beta2 for the Adam optimizer")
        parser.add_argument("--epochs", default=10, type=int)
        parser.add_argument("--lr", default=0.0001, type=float)
        parser.add_argument("--lr-scheduling", choices=("", "linear_with_warmup"), default="linear_with_warmup")
        parser.add_argument("--seed", type=int, default=42)
        model_name_choices = sorted([key
                                     for config in MODEL_FOR_PRETRAINING_MAPPING
                                     for key in config.pretrained_config_archive_map])
        parser.add_argument("--transformer-model-name", choices=model_name_choices, default="bert-base-uncased")
        parser.add_argument("--visual-size", default=500, type=int, metavar="V", help="input video feature dimension")
        parser.add_argument("--weight-decay", default=1e-4, type=float)
        parser.add_argument("--max-token-num", default=3, type=int,
                            help="max number of tokens predicted in val and test")
        return parser


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
    parent_parser.add_argument("--grad-eval", type=bool, default=False)
    parent_parser.add_argument("--mturk-eval", type=bool, default=False)
    parent_parser.add_argument("--mturk-data", type=str, default="")
    parser = MultiModalLightningModel.add_model_specific_args(parent_parser)
    return parser.parse_args()


def _main() -> None:
    hparams = _get_args()

    if hparams.seed is not None:
        _set_seed(hparams.seed)

    logging_level = logging.INFO if hparams.verbose else logging.WARNING
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging_level)

    logger.info(hparams)

    
    if hparams.grad_eval:
        model = MultiModalLightningModel.load_from_checkpoint(checkpoint_path='/home/ruoyaow/LifeQA-methodology/lightning_logs/version_6082788/checkpoints/epoch=0.ckpt')
        data = _dataloader('val2.pkl', hparams)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        for batch in data:
            batch_size = batch[0][0].shape[0]
            text_token_ids, visual, mask, segment_mask, labels, mask_positions, mask_lm_labels, position_ids = batch[0]
            out, embed = model(text_token_ids, visual, mask, segment_mask, mask_lm_labels, position_ids, True)
            loss, scores = out
            prediction_indices = torch.argmax(scores[list(range(batch_size)), mask_positions], dim=1)

            predictions = tokenizer.convert_ids_to_tokens(prediction_indices.tolist())
            correct = sum((len(label)==1 and prediction == label[0]) for prediction, label in zip(predictions, labels))

            loss.backward()
            embed_grad = embed.grad
            embed_sum = torch.abs(embed_grad).sum(axis=2)
            embed_sum_top3 = torch.topk(embed_sum, 3, axis=1)
            prediction_indices = torch.argmax(scores[list(range(batch_size)), mask_positions], dim=1)
            predictions = tokenizer.convert_ids_to_tokens(prediction_indices.tolist())
            for i in range(embed_grad.shape[0]):
                print(tokenizer.convert_ids_to_tokens(text_token_ids[i]))
                print(predictions[i])
                print(labels[i][0])
                print(embed_sum_top3[1][i])
            pass
    elif hparams.mturk_eval:
        # model = MultiModalLightningModel.load_from_checkpoint(checkpoint_path='/home/ruoyaow/LifeQA-methodology/great_lakes/lightning_logs/version_12390904/checkpoints/epoch=1.ckpt')
        model = MultiModalLightningModel.load_from_checkpoint(checkpoint_path='/home/ruoyaow/LifeQA-methodology/great_lakes/lightning_logs/version_8206545/checkpoints/epoch=1.ckpt')
        # model = AutoModelWithLMHead.from_pretrained('bert-base-uncased')
        model.eval()
        data = _dataloader(hparams.mturk_data, hparams)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        total = 0
        top10 = 0
        oot = 0
        correct_workers = 0
        correct_extended = 0
        correct_standard = 0

        for batch in data:
            batch_size = batch[0][0].shape[0]
            text_token_ids, visual, mask, segment_mask, labels, mask_positions, mask_lm_labels, position_ids, standard_answers, extended_answers = batch[0]
            # out = model(text_token_ids, visual, mask, segment_mask, mask_lm_labels, position_ids)
            scores = model(text_token_ids)[0]
            # loss, scores = out
            prediction_indices = torch.argmax(scores[list(range(batch_size)), mask_positions], dim=1)

            predictions = tokenizer.convert_ids_to_tokens(prediction_indices.tolist())
            worker_results = [(prediction in label) for prediction, label in zip(predictions, labels)]
            extended_results = [(prediction in label) for prediction, label in zip(predictions, extended_answers)]
            standard_results = [(prediction == label) for prediction, label in zip(predictions, standard_answers)]
            
            
            
            for i in range(batch_size):
                _, indices = torch.topk(scores[i,mask_positions[i]], 10)
                predictions_top10 = tokenizer.convert_ids_to_tokens(indices)
                correct = False
                
                oot_score = 0
                total_H = 0
                if labels[i] != []:
                    for key in labels[i]:
                        total_H += labels[i][key][0] / labels[i][key][1]

                    for pred in predictions_top10:
                        if pred in labels[i]:
                            correct = True
                            oot_score += labels[i][pred][0] / labels[i][pred][1]
                
                    oot_score /= total_H
                else:
                    for pred in predictions_top10:
                        if pred == standard_answers[i]:
                            correct = True
                            oot_score = 1
                            break
                if correct:
                    top10 += 1
                oot += oot_score


            # for i in range(len(extended_results)):
            #     if extended_results[i] != standard_results[i]:
            #         print(' '.join(tokenizer.convert_ids_to_tokens(text_token_ids[i].tolist())))
            #         print(predictions[i])
            #         print(labels[i])
            #         print(standard_answers[i])
            #         print()
            correct_workers += sum(worker_results)
            correct_extended += sum(extended_results)
            correct_standard += sum(standard_results)
            total += batch_size

        acc_top10 = top10 / total
        oot = oot / total
        acc_workers = correct_workers / total
        acc_extended = correct_extended / total
        acc_standard = correct_standard / total
        print(acc_top10)
        print(oot)
        print(acc_workers)
        print(acc_extended)
        print(acc_standard)
    else:
        model = MultiModalLightningModel(hparams)

        trainer = pl.Trainer(default_root_dir=hparams.save_path, num_gpus=hparams.gpu_count, max_epochs=hparams.epochs,
                            distributed_backend=hparams.distributed_backend, use_amp=hparams.use_16bit, benchmark=True,
                            amp_level=hparams.amp_level, resume_from_checkpoint=hparams.resume_from_checkpoint,
                            progress_bar_refresh_rate=1, overfit_pct=hparams.overfit_pct,
                            fast_dev_run=hparams.fast_dev_run)

        trainer.fit(model)
            


if __name__ == "__main__":
    _main()
