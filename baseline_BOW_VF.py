import logging
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pickle
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils import build_representation, fit
from data_loader_multimodal import ActivityNetCaptionsDataset
from pytorch_lightning import Trainer
from argparse import ArgumentParser


class BaselineBowVF(pl.LightningModule):

    def __init__(self, hparams, idx_to_word_map, word_to_idx_map, output_dim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(BaselineBowVF, self).__init__()

        self.word_to_idx = word_to_idx_map
        self.idx_to_word = idx_to_word_map
        self.num_vocabs = output_dim
        
        self.hparams = hparams

        self.linear1 = nn.Linear(self.hparams.text_feature_dim, self.hparams.video_feature_dim)
        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(2 * self.hparams.video_feature_dim, output_dim)

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
        text_feature, video_feature, labels, _, _ = train_batch
        # labels shape: [batch_size, 1] --> [batch_size]
        labels = labels.squeeze(dim=1)

        logits = self.forward(text_feature, video_feature)
        loss = self.cross_entropy(logits, labels)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        text_feature, video_feature, labels, video_ids, questions = val_batch
        labels = labels.squeeze(dim=1)

        logits = self.forward(text_feature, video_feature)
        loss = self.cross_entropy(logits, labels)

        # calculate accuracy
        correct_predictions, batch_size = 0, len(labels)
        metric1_predictions, vqa1_predictions = 0, 0
        for i in range(batch_size):
            video_id, question = video_ids[i], questions[i]
            pred = torch.argmax(logits[i])
            if pred == labels[i]:
                correct_predictions += 1
            # mturk answer check
            pred_ans = self.idx_to_word[pred.item()]
            worker_ans = self.mturk_val_answers[video_id][question]['worker_answers']
            if worker_ans.get(pred_ans):
                # metric 1
                most_freq_ans_count = max([freq for freq, _ in worker_ans.values()])
                pred_freq_ans_count = worker_ans[pred_ans][0]
                metric1_predictions += pred_freq_ans_count / most_freq_ans_count
                # vqa1
                vqa1_predictions += 1

        # logging training loss
        logs = {
            'val_loss': loss,
            'validation_accuracy': torch.tensor([correct_predictions / batch_size]),
            'metric1_val_accuracy': torch.tensor([metric1_predictions / batch_size]),
            'vqa1_val_accuracy': torch.tensor([vqa1_predictions / batch_size])
        }
        return logs

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is [{'loss': batch_0_loss, 'loss': batch_1_loss, ...}]
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['validation_accuracy'] for x in outputs]).mean()
        avg_metric1_accuracy = torch.stack([x['metric1_val_accuracy'] for x in outputs]).mean()
        avg_vqa1_accuracy = torch.stack([x['vqa1_val_accuracy'] for x in outputs]).mean()

        tensorboard_logs = {
            'val_loss': avg_loss,
            'validation_accuracy': avg_accuracy,
            'metric1_val_accuracy': avg_metric1_accuracy,
            'vqa1_val_accuracy': avg_vqa1_accuracy
        }

        return {'avg_val_loss': avg_loss, 'validation_accuracy': avg_accuracy, 'log': tensorboard_logs}

    # def test_step(self, test_batch, batch_idx):
    #     text_feature, video_feature, labels, video_ids, questions = test_batch
    #     logits = self.forward(text_feature, video_feature)
    #     loss = self.cross_entropy(logits, labels)
    #
    #     correct_predictions, batch_size = 0, len(labels)
    #     metric1_predictions, vqa1_predictions = 0, 0
    #
    #     for i in range(batch_size):
    #         video_id, question = video_ids[i], questions[i]
    #         pred = torch.argmax(logits[i])
    #         if pred == labels[i]:
    #             correct_predictions += 1
    #         # # mturk answer check
    #         pred_ans = self.idx_to_word[pred.item()]
    #         worker_ans = self.mturk_test_answers[video_id][question]['worker_answers']
    #         if worker_ans.get(pred_ans):
    #             # metric 1
    #             most_freq_ans_count = max([freq for freq, _ in worker_ans.values()])
    #             pred_freq_ans_count = worker_ans[pred_ans][0]
    #             metric1_predictions += pred_freq_ans_count / most_freq_ans_count
    #             # vqa1
    #             vqa1_predictions += 1
    #
    #     logs = {
    #         'test_loss': loss,
    #         'test_accuracy': torch.tensor([correct_predictions / batch_size]),
    #         'metric1_test_accuracy': torch.tensor([metric1_predictions / batch_size]),
    #         'vqa1_test_accuracy': torch.tensor([vqa1_predictions / batch_size])
    #     }
    #     return logs
    #
    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     avg_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()
    #     avg_metric1_accuracy = torch.stack([x['metric1_test_accuracy'] for x in outputs]).mean()
    #     avg_vqa1_accuracy = torch.stack([x['vqa1_test_accuracy'] for x in outputs]).mean()
    #     tensorboard_logs = {
    #         'test_loss': avg_loss,
    #         'test_accuracy': avg_accuracy,
    #         'metric1_test_accuracy': avg_metric1_accuracy,
    #         'vqa1_test_accuracy': avg_vqa1_accuracy
    #     }
    #
    #     return {'test_loss': avg_loss, 'test_accuracy': avg_accuracy, 'log': tensorboard_logs}
        
    def prepare_data(self):
        # load mturk worker answers
        mturk_val_path = os.path.join(self.hparams.data_path, 'latest_data/mturk_val.pickle')
        mturk_test_path = os.path.join(self.hparams.data_path, 'latest_data/mturk_test.pickle')
        self.mturk_val_answers = pickle.load(open(mturk_val_path, 'rb'))
        self.mturk_test_answers = pickle.load(open(mturk_test_path, 'rb'))
        
        # load video duration variable from file
        duration_path = os.path.join(self.hparams.data_path, 'latest_data/multimodal_model/video_duration.pkl')
        self.video_duration = pickle.load(open(duration_path, 'rb'))
        # build representations

        video_features_file = os.path.join(self.hparams.data_path,
                                           'ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5')

        bow_representation_train_file = os.path.join(self.hparams.data_path, 'latest_data/bow_training_data.pkl')
        bow_representation_val_file = os.path.join(self.hparams.data_path, 'latest_data/bow_validation_data.pkl')
        bow_representation_test_file = os.path.join(self.hparams.data_path, 'latest_data/bow_test_data.pkl')

        masked_training_data_file = os.path.join(self.hparams.data_path, 'latest_data/train')
        masked_validation_data_file = os.path.join(self.hparams.data_path, 'latest_data/val')
        masked_test_data_file = os.path.join(self.hparams.data_path, 'latest_data/test')

        build_representation(masked_training_data_file, video_features_file,
                             bow_representation_train_file, self.video_duration,
                             self.word_to_idx, self.num_vocabs)
        build_representation(masked_validation_data_file, video_features_file,
                             bow_representation_val_file, self.video_duration,
                             self.word_to_idx, self.num_vocabs)
        build_representation(masked_test_data_file, video_features_file,
                             bow_representation_test_file, self.video_duration,
                             self.word_to_idx, self.num_vocabs)

        self.training_data_set = ActivityNetCaptionsDataset(bow_representation_train_file)
        self.validation_data_set = ActivityNetCaptionsDataset(bow_representation_val_file)
        self.test_data_set = ActivityNetCaptionsDataset(bow_representation_test_file)

    def train_dataloader(self):
        return DataLoader(self.training_data_set, batch_size=self.hparams.train_batch_size,
                          shuffle=self.hparams.shuffle, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_data_set, batch_size=self.hparams.val_batch_size,
                          shuffle=self.hparams.shuffle, num_workers=self.hparams.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.test_data_set, batch_size=self.hparams.test_batch_size,
    #                       shuffle=self.hparams.shuffle, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--weight_decay", default=0, type=float)
        parser.add_argument('--num_tokens', type=int, default=1000)
        parser.add_argument("--epochs", type=int, default=1000)
        parser.add_argument('--text_feature_dim', type=int, default=1000)
        parser.add_argument('--video_feature_dim', type=int, default=500)
        parser.add_argument('--shuffle', type=bool, default=True)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--train_batch_size', type=int, default=32)
        parser.add_argument('--val_batch_size', type=int, default=32)
        parser.add_argument('--test_batch_size', type=int, default=32)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        return parser

    @staticmethod
    # program level args
    def add_args():
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

        return BaselineBowVF.add_model_specific_args(parser)


if __name__ == '__main__':
    parser = BaselineBowVF.add_args()
    hparams = parser.parse_args()

    train_text_file = os.path.join(hparams.data_path, 'latest_data/train.json')
    idx_to_word, word_to_idx, NUM_VOCABS = fit(train_text_file, hparams.num_tokens)

    parser = Trainer.add_argparse_args(parser)

    logger = logging.getLogger(__name__)
    logger.info(hparams)

    logging.basicConfig(level=hparams.loglevel)

    trainer = Trainer(default_save_path=hparams.save_path, gpus=hparams.gpu_count, max_epochs=hparams.epochs,
                      distributed_backend=hparams.distributed_backend, benchmark=True,
                      resume_from_checkpoint=hparams.resume_from_checkpoint, fast_dev_run=hparams.fast_dev_run,
                      progress_bar_refresh_rate=1)

    model = BaselineBowVF(hparams, idx_to_word, word_to_idx, NUM_VOCABS)
    trainer.fit(model)
