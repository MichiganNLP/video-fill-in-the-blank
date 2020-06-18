import logging
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils import build_representation, fit
from data_loader_multimodal import ActivityNetCaptionsDataset
from pytorch_lightning import Trainer
from argparse import ArgumentParser


class BaselineBowVF(pl.LightningModule):

    def __init__(self, hparams):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(BaselineBowVF, self).__init__()

        self.hparams = hparams

        self.linear1 = nn.Linear(self.hparams.text_feature_dim, self.hparams.video_feature_dim)
        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(2 * self.hparams.video_feature_dim, self.hparams.output_dim)

    def forward(self, text_feature, video_feature):

        text_feature_out = self.linear1(text_feature)
        text_feature_out = self.relu(text_feature_out)

        fused_feature = torch.cat([text_feature_out, video_feature], dim=-1)

        y_pred = self.linear2(fused_feature)

        y_pred = torch.log_softmax(y_pred, dim=1)

        return y_pred

    @staticmethod
    def cross_entropy(logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        text_feature, video_feature, labels = train_batch
        # labels shape: [batch_size, 1] --> [batch_size]
        labels = labels.squeeze(dim=1)

        logits = self.forward(text_feature, video_feature)
        loss = self.cross_entropy(logits, labels)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        text_feature, video_feature, labels = val_batch
        labels = labels.squeeze(dim=1)

        logits = self.forward(text_feature, video_feature)
        loss = self.cross_entropy(logits, labels)

        # calculate accuracy
        correct_predictions, batch_size = 0, len(val_batch)
        for i in range(batch_size):
            if torch.argmax(logits[i]) == labels[i]:
                correct_predictions += 1

        # logging training loss
        logs = {'val_loss': loss, 'validation_accuracy': torch.tensor([correct_predictions / batch_size])}
        return logs

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is [{'loss': batch_0_loss, 'loss': batch_1_loss, ...}]
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['validation_accuracy'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'validation_accuracy': avg_accuracy}

        return {'avg_val_loss': avg_loss, 'validation_accuracy': avg_accuracy, 'log': tensorboard_logs}

    def prepare_data(self):
        # build representations

        video_features_file = os.path.join(self.hparams.data_path,
                                           'ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5')
        train_text_file = os.path.join(self.hparams.data_path, 'train.json')

        bow_representation_train_file = os.path.join(self.hparams.data_path, 'bow_training_data.pkl')
        bow_representation_val_file_1 = os.path.join(self.hparams.data_path, 'bow_validation_data_1.pkl')
        bow_representation_val_file_2 = os.path.join(self.hparams.data_path, 'bow_validation_data_2.pkl')

        masked_training_data_file = os.path.join(self.hparams.data_path, 'train')
        masked_validation_data_file_1 = os.path.join(self.hparams.data_path, 'val1')
        masked_validation_data_file_2 = os.path.join(self.hparams.data_path, 'val2')

        build_representation(self.hparams.num_tokens, masked_training_data_file, train_text_file,
                             video_features_file, bow_representation_train_file)
        build_representation(self.hparams.num_tokens, masked_validation_data_file_1, train_text_file,
                             video_features_file, bow_representation_val_file_1)
        build_representation(self.hparams.num_tokens, masked_validation_data_file_2, train_text_file,
                             video_features_file, bow_representation_val_file_2)

        self.training_data_set = ActivityNetCaptionsDataset(bow_representation_train_file)
        self.validation_data_set = ActivityNetCaptionsDataset(bow_representation_val_file_1)
        self.test_data_set = ActivityNetCaptionsDataset(bow_representation_val_file_2)

    def train_dataloader(self):
        return DataLoader(self.training_data_set, batch_size=self.hparams.train_batch_size,
                          shuffle=self.hparams.shuffle, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_data_set, batch_size=self.hparams.val_batch_size,
                          shuffle=self.hparams.shuffle, num_workers=self.hparams.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.test_data_set, batch_size=16, shuffle=True, num_workers=0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser, num_vocabs, num_tokens):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_tokens', type=int, default=num_tokens)
        parser.add_argument("--epochs", type=int, default=1000)
        parser.add_argument('--text_feature_dim', type=int, default=1000)
        parser.add_argument('--video_feature_dim', type=int, default=500)
        parser.add_argument('--output_dim', type=int, default=num_vocabs)
        parser.add_argument('--shuffle', type=bool, default=True)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--train_batch_size', type=int, default=32)
        parser.add_argument('--val_batch_size', type=int, default=16)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        return parser

    @staticmethod
    # program level args
    def add_program_args():
        parser = ArgumentParser(add_help=False)
        parser.add_argument(
            '-d', '--debug',
            help="debugging mode",
            action="store_const", dest="loglevel", const=logging.DEBUG,
            default=logging.WARNING,
        )
        parser.add_argument(
            '-v', '--verbose',
            help="verbose mode",
            action="store_const", dest="loglevel", const=logging.INFO,
        )
        parser.add_argument('--fast-dev-run', action='store_true')
        parser.add_argument('--data-path', type=str, default='data')
        parser.add_argument("--save-path", metavar="DIR", default="./")
        parser.add_argument('--distributed-backend', default='dp', choices=('dp', 'ddp', 'ddp2'))
        parser.add_argument('--gpu-count', type=int, default=1)
        parser.add_argument('--resume-from-checkpoint', metavar='CHECKPOINT_FILE')
        return parser


if __name__ == '__main__':
    parser = BaselineBowVF.add_program_args()
    train_text_file = os.path.join(parser.parse_args().data_path, 'train.json')

    NUM_TOKENS = 1000
    _, NUM_VOCABS = fit(train_text_file, NUM_TOKENS)

    parser = BaselineBowVF.add_model_specific_args(parser, NUM_VOCABS, NUM_TOKENS)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info(hparams)

    logging.basicConfig(level=hparams.loglevel)

    trainer = Trainer(default_save_path=hparams.save_path, gpus=hparams.gpu_count, max_epochs=hparams.epochs,
                      distributed_backend=hparams.distributed_backend, benchmark=True,
                      resume_from_checkpoint=hparams.resume_from_checkpoint, fast_dev_run=hparams.fast_dev_run,
                      progress_bar_refresh_rate=1)

    model = BaselineBowVF(hparams)
    trainer.fit(model)
