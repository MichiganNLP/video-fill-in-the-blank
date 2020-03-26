import argparse
import os
import random
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
from data_loader_multimodal import ActivityNetCaptionDataset
from utils import batchPadding
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule

class MultiModalLightningModel(LightningModule):
    def __init__(self, hparams):

        super(MultiModalLightningModel, self).__init__()
        self.hparams = hparams
        self.video_embedding = nn.Linear(self.hparams.V_D_in, self.hparams.embedding_size)
        self.text_embedding = BertModel.from_pretrained('bert-base-uncased').get_input_embeddings()
        self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=False)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, text_feature, video_feature, attention_mask, segment_mask, mask_lm_labels, position_embedding):
        video_feature_embeddings = self.video_embedding(video_feature)
        text_feature_embeddings = self.text_embedding(text_feature)

        input_feature = torch.cat([text_feature_embeddings, video_feature_embeddings], dim=1)
        out = self.bert_model(inputs_embeds=input_feature, attention_mask=attention_mask, token_type_ids=segment_mask,masked_lm_labels=mask_lm_labels, position_ids=position_embedding)
        return out

    def training_step(self, batch, batch_idx):
        textFeatures, videoFeatures, attention_mask, segment_mask, labels, mask_positions, mask_lm_labels, position_embedding, _ = batch
        # if torch.cuda.is_available():
        #     textFeatures = textFeatures.cuda()
        #     videoFeatures = videoFeatures.cuda()
        #     attention_mask = attention_mask.cuda()
        #     segment_mask = segment_mask.cuda()
        #     mask_lm_labels = mask_lm_labels.cuda()
        #     position_embedding = position_embedding.cuda()
        
        output = self.forward(textFeatures, videoFeatures, attention_mask, segment_mask, mask_lm_labels,position_embedding)
        loss_val = output[0]
        acc1 = self.__accuracy(textFeatures, output[1], mask_positions)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'acc1': acc1,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output

    def validation_step(self, batch, batch_idx):
        textFeatures, videoFeatures, attention_mask, segment_mask, labels, mask_positions, mask_lm_labels, position_embedding, key = batch
        # if torch.cuda.is_available():
        #     textFeatures = textFeatures.cuda()
        #     videoFeatures = videoFeatures.cuda()
        #     attention_mask = attention_mask.cuda()
        #     segment_mask = segment_mask.cuda()
        #     mask_lm_labels = mask_lm_labels.cuda()
        #     position_embedding = position_embedding.cuda()   
        
        output = self.forward(textFeatures, videoFeatures, attention_mask, segment_mask, mask_lm_labels, position_embedding)
        loss_val = output[0]
        acc1 = self.__accuracy(textFeatures, output[1], mask_positions)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc1': acc1,
        })

        return output

    def validation_epoch_end(self, outputs):

        tqdm_dict = {}

        for metric_name in ["val_loss", "val_acc1"]:
            metric_total = 0

            for output in outputs:
                metric_value = output[metric_name]

                metric_total += metric_value

            tqdm_dict[metric_name] = metric_total / len(outputs)

        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
        return result

    @classmethod
    def __accuracy(cls, textFeatures, score, mask_positions):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            correct = 0
            total_num = 0
            batch_size = textFeatures.shape[0]
            predicted_index = torch.argmax(score[list(range(batch_size)), mask_positions], dim=1)

            top5=score[list(range(batch_size)), mask_positions].topk(5, dim=1)[1]

            out_text = self.tokenizer.convert_ids_to_tokens(predicted_index.tolist())
            total_num += batch_size
            for i in range(batch_size):
                if labels[i] == out_text[i]:
                    correct += 1
            return correct / total_num

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * self.hparams.epochs, self.hparams.epochs)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        trainDataset = ActivityNetCaptionDataset(self.hparams.data_path)
        train_loader = DataLoader(trainDataset, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=batchPadding, num_workers=self.hparams.num_workers)
        return train_loader

    def val_dataloader(self):
        valDataset = ActivityNetCaptionDataset(self.hparams.data_path)
        val_loader = DataLoader(valDataset, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=batchPadding, num_workers=self.hparams.num_workers)
        return val_loader

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--epochs', default=10, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=42,
                            help='seed for initializing training. ')
        parser.add_argument('-b', '--batch_size', default=16, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--beta1', default=0.9, type=float, metavar='M1',
                            help='beta1 for adam optimizer')
        parser.add_argument('--beta2', default=0.999, type=float, metavar='M2',
                            help='beta2 for adam optimizer')
        parser.add_argument('--V_D_in', default=500, type=int, metavar='V',
                            help='input video feature dimension')
        parser.add_argument('--embedding_size', default=768, type=float, metavar='E',
                            help='word embedding size')
        parser.add_argument('--num_workers', default=8, type=float, 
                            help='number of workers used for data loading')
        parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        return parser


def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data_path', metavar='DIR', type=str,
                               help='path to dataset')
    parent_parser.add_argument('--save-path', metavar='DIR', default=".", type=str,
                               help='path to save output')
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='how many gpus')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--use-16bit', dest='use_16bit', action='store_true',
                               help='if true uses 16 bit precision')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')

    parser = MultiModalLightningModel.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(hparams):
    model = MultiModalLightningModel(hparams)
    if hparams.seed is not None:
        random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
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