import pickle

from lqam.methods.dataset import QGenDataModule
from lqam.methods.metrics import ExactMatchAccuracyMany, F1ScoreMany
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer

import argparse
from lqam.methods import dataset
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path
import json


class MostSimilarModule():
    def __init__(self, glove_vectors_path, output_path):
        with open(glove_vectors_path, 'rb') as f:
            self.glove_vectors = pickle.load(f)
        self.output_path = output_path

        self.similarity = nn.CosineSimilarity()
        self.accuracy = ExactMatchAccuracyMany()
        self.f1score = F1ScoreMany()
        self.accuracy_many = ExactMatchAccuracyMany()
        self.f1score_many = F1ScoreMany()


    def find_most_similar(self, feature):
        min_distance = 1 # cosine distance range 0-1
        answer = None
        for word in self.glove_vectors:
            distance = self.similarity(feature, self.glove_vectors[word])
            if distance < min_distance:
                min_distance = distance
                answer = word

        return answer

    def get_feature(self, sentence):
        word_list = sentence.split()
        vectors = []
        vector_size = self.glove_vectors['the'].shape[1]
        for word in word_list:
            if word in self.glove_vectors:
                vectors.append(self.glove_vectors[word])
            elif word == '_____':
                continue
            else:
                vectors.append(torch.zeros(1, vector_size))
        
        feature = torch.mean(torch.cat(vectors), dim=0, keepdim=True)
        return feature


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

            masked_caption_feature = self.get_feature(masked_caption)

            answer = self.find_most_similar(masked_caption_feature)
            self.accuracy([answer], [label])
            self.f1score([answer], [label])
            self.accuracy_many([answer], [label], [additional_answers])
            self.f1score_many([answer], [label], [additional_answers])
            pred.append([masked_caption, answer, label])

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
        if prefix == 'test':
            df = pd.DataFrame(pred, columns=['masked_caption', 'generated', 'ground_truth'])
            df.to_csv(self.output_path, index=False)
            print(f"Predictions saved in {self.output_path}. First rows:")
            print()
            pd.options.display.float_format = self._pandas_float_format
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 0,
                                "display.max_colwidth", None):
                print(df.head(10))


    def val(self, val_data):
        pred, accuracy, f1score, accuracy_many, f1score_many = self.val_test_step(val_data)
        self.output('val', pred, accuracy, f1score, accuracy_many, f1score_many)


    def test(self, test_data):
        pred, accuracy, f1score, accuracy_many, f1score_many = self.val_test_step(test_data)
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
    parser.add_argument("--test-data-path", default=dataset.URL_DATA_VAL)  # TODO: change to test.
    parser.add_argument("--glove_vectors_path", default="data/GloVe/glove_embed.pkl")

    parser.add_argument("--num-workers", "-j", type=int, default=0,
                        help="data loader workers. Each worker batch-tokenizes in parallel, "
                             "so maybe don't make this number equal to the number of CPU cores but just a small "
                             "natural number.")

    parser.add_argument("--model", default="t5-base",
                        help="pipeline model. Check the options in https://huggingface.co/models?filter=seq2seq")

    parser.add_argument("--predictions-output-path", default="predictions_most_similar_glove.csv")

    return parser.parse_args()

def load_data(data_path: str):
    with open(cached_path(data_path)) as file:
        instances = json.load(file)

    output = []
    count = 0
    for instance in instances:
        if count >= 10:
            break
        count += 1
        output.append({
            "masked_caption": instance["masked_caption"],
            "label": instance["label"],
            "additional_answers": instance["additional_answers"]
        })

    return output

args = _parse_args()

model = MostSimilarModule(args.glove_vectors_path, args.predictions_output_path)

model.val(load_data(args.val_data_path))
model.test(load_data(args.test_data_path))
