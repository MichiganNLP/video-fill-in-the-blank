import argparse
from collections import defaultdict

from torch.utils.data import DataLoader

from lqam.methods import dataset
from lqam.methods.dataset import QGenDataModule
from lqam.methods.metrics import AllMetrics
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults


class MostFreqModule:
    def __init__(self) -> None:
        self.most_freq_ans = None
        self.all_metrics = AllMetrics(compute_prob=False)

    def train(self, data_loader: DataLoader) -> None:
        freq = defaultdict(int)
        max_freq = 0
        most_freq_ans = None

        for data in data_loader:
            # TODO: consider all the answers.
            label = data['label'][0].lower()  # TODO: Do all the normalization.
            freq[label] += 1

            if freq[label] > max_freq:
                max_freq = freq[label]
                most_freq_ans = label

        self.most_freq_ans = most_freq_ans

    def eval(self, data_loader: DataLoader, prefix: str) -> None:
        self.all_metrics.reset()

        for data in data_loader:
            self.all_metrics(data['video_id'], data['label'], data['additional_answers'], [self.most_freq_ans])

        for k, v in self.all_metrics.compute().items():
            print(f"{prefix}_{k}", v)


def _parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults(description="Train and evaluate the most similar baseline.")
    parser.add_argument("--train-data-path", default=dataset.URL_DATA_TRAIN)
    parser.add_argument("--val-data-path", default=dataset.URL_DATA_VAL)
    parser.add_argument("--test-data-path", default=dataset.URL_DATA_TEST)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = MostFreqModule()
    data_module = QGenDataModule(batch_size=1, train_data_path=args.train_data_path,
                                 val_data_path=args.val_data_path, test_data_path=args.test_data_path)
    model.train(data_module.train_dataloader())
    model.eval(data_module.val_dataloader(), "val")
    model.eval(data_module.test_dataloader(), "test")


if __name__ == '__main__':
    main()
