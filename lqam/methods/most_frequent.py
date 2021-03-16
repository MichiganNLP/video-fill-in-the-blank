import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from lqam.methods.dataset import QGenDataModule
from lqam.methods.metrics import ExactMatchAccuracyMany, F1ScoreMany
import pandas as pd

import argparse
from lqam.methods import dataset
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults


class MostFreqModule():
    def __init__(self, model, output_path):
        self.most_freq_ans = ''

        self.accuracy = ExactMatchAccuracyMany()
        self.f1score = F1ScoreMany()
        self.accuracy_many = ExactMatchAccuracyMany()
        self.f1score_many = F1ScoreMany()
        self.output_path = output_path 
        

    def train(self, train_data_loader):
        freq = {}
        max_freq = 0
        most_freq_ans = ''

        for data in train_data_loader:

            label = data['label'][0].lower()
            if label in freq:
                freq[label] += 1
            else:
                freq[label] = 1

            if freq[label] > max_freq:
                max_freq = freq[label]
                most_freq_ans = label

        self.most_freq_ans = most_freq_ans       

    def val_test_step(self, data_loader):
        self.accuracy.reset()
        self.f1score.reset()
        self.accuracy_many.reset()
        self.f1score_many.reset()
        pred = []
        for data in data_loader:
            masked_caption = data['masked_caption']
            label = data['label']
            additional_answers = data['additional_answers']


            self.accuracy([self.most_freq_ans], label)
            self.f1score([self.most_freq_ans], label)
            self.accuracy_many([self.most_freq_ans], label, additional_answers)
            self.f1score_many([self.most_freq_ans], label, additional_answers)
            pred.append([masked_caption[0], self.most_freq_ans, label[0]])

        accuracy = self.accuracy.compute()
        f1score = self.f1score.compute()
        accuracy_many = self.accuracy_many.compute()
        f1score_many = self.f1score_many.compute()
        return pred, accuracy, f1score, accuracy_many, f1score_many
                
    def output(self, prefix, pred, accuracy, f1score, accuracy_many, f1score_many):
        print(f'{prefix}_accuracy:', accuracy.item())
        print(f'{prefix}_f1score:', f1score.item())
        print(f'{prefix}_accuracy_many:', accuracy_many.item())
        print(f'{prefix}_f1score_many:', f1score_many.item())
        # if prefix == 'test':
        #     df = pd.DataFrame(pred, columns=['masked_caption', 'generated', 'ground_truth'])
        #     df.to_csv(self.output_path, index=False)
        #     print(f"Predictions saved in {self.output_path}. First rows:")
        #     print()
        #     pd.options.display.float_format = self._pandas_float_format
        #     with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 0,
        #                         "display.max_colwidth", None):
        #         print(df.head(10))


    def val(self, val_data_loader):
        pred, accuracy, f1score, accuracy_many, f1score_many = self.val_test_step(val_data_loader)
        self.output('val', pred, accuracy, f1score, accuracy_many, f1score_many)


    def test(self, test_data_loader):
        pred, accuracy, f1score, accuracy_many, f1score_many = self.val_test_step(test_data_loader)
        self.output('test', pred, accuracy, f1score, accuracy_many, f1score_many)

    @staticmethod
    def _pandas_float_format(x: float) -> str:
        if x == 0:
            return "0"
        elif abs(x) < 0.005:
            return np.format_float_scientific(x, exp_digits=1, precision=0, trim="-")
        else:
            return f"{x:.2f}"

def _parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults(description="Train and evaluate the most similar baseline.")

    parser.add_argument("--train-data-path", default=dataset.URL_DATA_TRAIN)
    parser.add_argument("--val-data-path", default=dataset.URL_DATA_VAL)
    parser.add_argument("--test-data-path", default=dataset.URL_DATA_TEST)  # TODO: change to test.
    parser.add_argument("--visual-data-dir", default="data/I3D_video_features")

    parser.add_argument("--num-workers", "-j", type=int, default=0,
                        help="data loader workers. Each worker batch-tokenizes in parallel, "
                             "so maybe don't make this number equal to the number of CPU cores but just a small "
                             "natural number.")

    parser.add_argument("--model", default="t5-base",
                        help="pipeline model. Check the options in https://huggingface.co/models?filter=seq2seq")

    parser.add_argument("--predictions-output-path", default="predictions.csv")

    parser.add_argument("--trainer-default-root-dir")

    return parser.parse_args()



args = _parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = MostFreqModule(args.model, args.predictions_output_path)
data_module = QGenDataModule(tokenizer=tokenizer, batch_size=1, num_workers=args.num_workers,
                                 train_data_path=args.train_data_path, val_data_path=args.val_data_path,
                                 test_data_path=args.test_data_path, visual_data_dir=None)

model.train(data_module.train_dataloader())
model.val(data_module.val_dataloader())
model.test(data_module.test_dataloader())