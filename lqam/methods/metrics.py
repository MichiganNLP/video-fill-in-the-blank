from typing import Iterable, Iterator, Optional, Sequence

import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from lqam.util.file_utils import cached_path
from lqam.core.metrics import compute_token_level_f1_many, exact_match_many, flatten_all_answers, \
    tokenize_answer_to_compute_metrics


class F1ScoreMany(pl.metrics.Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def _tokenize_answers(answers: Iterable[str]) -> Iterator[Iterator[str]]:
        yield from (tokenize_answer_to_compute_metrics(answer) for answer in answers)

    @overrides
    def update(self, preds: Sequence[str], labels: Sequence[str],
               additional_answers_batch: Optional[Sequence[Sequence[Sequence[str]]]] = None) -> None:
        assert len(preds) == len(labels)
        answers_batch = flatten_all_answers(labels, additional_answers_batch)
        self.score_sum += sum(compute_token_level_f1_many(tokenize_answer_to_compute_metrics(pred),
                                                          self._tokenize_answers(answers))
                              for pred, answers in zip(preds, answers_batch))
        self.total += len(labels)

    @overrides
    def compute(self) -> torch.Tensor:
        return self.score_sum / self.total


class ExactMatchAccuracyMany(pl.metrics.Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @overrides
    def update(self, preds: Sequence[str], labels: Sequence[str],
               additional_answers_batch: Optional[Sequence[Sequence[Sequence[str]]]] = None) -> None:  # noqa
        assert len(preds) == len(labels)
        answers_batch = flatten_all_answers(labels, additional_answers_batch)
        self.correct += sum(exact_match_many(pred, answers) for pred, answers in zip(preds, answers_batch))
        self.total += len(labels)

    @overrides
    def compute(self) -> torch.Tensor:
        return self.correct / self.total


class Average(pl.metrics.Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Accumulate values to lose less precision.
        self.add_state("values", default=[], dist_reduce_fx=None)

    @overrides
    def update(self, t: torch.Tensor) -> None:  # noqa
        self.values.append(t)

    @overrides
    def compute(self) -> torch.Tensor:
        return torch.cat(self.values, dim=0).mean()


# From https://github.com/pytorch/pytorch/issues/21987#issuecomment-539402619
def nanmean(v: torch.Tensor, *args, inplace: bool = False, **kwargs) -> torch.Tensor:
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


class Perplexity(pl.metrics.Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Accumulate values to lose less precision.
        self.add_state("answer_probs", default=[], dist_reduce_fx=None)
        self.add_state("mask", default=[], dist_reduce_fx=None)

    @overrides
    def update(self, answer_probs: torch.Tensor, mask: torch.Tensor) -> None:  # noqa
        self.answer_probs.extend(answer_probs)
        self.mask.extend(mask)

    @overrides
    def compute(self) -> torch.Tensor:
        answer_probs = pad_sequence(self.answer_probs, batch_first=True, padding_value=1)
        mask = pad_sequence(self.mask, batch_first=True)

        answer_probs[~mask] = float("NaN")

        return torch.exp2(-nanmean(torch.log2(answer_probs), dim=1)).mean()

class ComputeMetrics():
    def __init__(self, category_file_path: str, *args, **kwargs) -> None:

        self.em = ExactMatchAccuracyMany()
        self.f1_score = F1ScoreMany()
        self.em_label = ExactMatchAccuracyMany()
        self.f1_score_label = F1ScoreMany()
        
        self.label_category = self.preprocess_category(category_file_path)
        self.em_cat = [ExactMatchAccuracyMany() for i in range(11)]
        self.f1_cat = [F1ScoreMany() for i in range(11)]
        self.em_label_cat = [ExactMatchAccuracyMany() for i in range(11)]
        self.f1_label_cat = [F1ScoreMany() for i in range(11)]

        self.gt_prob = Average()
        self.perplexity = Perplexity()

    def preprocess_category(self, category_file_path):
        cat_df = pd.read_csv(cached_path(category_file_path), sep="\t")

        output = {}
        for idx, row in cat_df.iterrows():
            output[row['video_id']] = row['category']

        return output

    def reset(self):
        self.em.reset()
        self.f1_score.reset()
        self.em_label.reset()
        self.f1_score_label.reset()
        
        for i in range(11):
            self.em_cat[i].reset()
            self.f1_cat[i].reset()
            self.em_label_cat[i].reset()
            self.f1_label_cat[i].reset()
        self.gt_prob.reset()
        self.perplexity.reset()

    def update(self, preds: Sequence[str], video_ids: Sequence[str], labels: Sequence[str],
                 additional_answers_batch: Sequence[Sequence[Sequence[str]]],
                 label_prob: torch.Tensor,
                 label_probs: torch.Tensor, perplexity_mask: torch.Tensor) -> None:
        self.em(preds, labels, additional_answers_batch)
        self.f1_score(preds, labels, additional_answers_batch)
        self.em_label(preds, labels)
        self.f1_score_label(preds, labels)

        for i in range(len(preds)):
            category = self.label_category[video_ids[i]]
            self.em_cat[category]([preds[i]], [labels[i]], [additional_answers_batch[i]])
            self.f1_cat[category]([preds[i]], [labels[i]], [additional_answers_batch[i]])
            self.em_label_cat[category]([preds[i]], [labels[i]])
            self.f1_label_cat[category]([preds[i]], [labels[i]])

        self.gt_prob(label_prob)
        self.perplexity(label_probs, perplexity_mask)

    def compute(self):
        em = self.em.compute()
        f1_score = self.f1_score.compute()
        em_label = self.em_label.compute()
        f1_score_label = self.f1_score_label.compute()

        em_cat = [self.em_cat[i].compute() for i in range(11)]
        f1_cat = [self.f1_cat[i].compute() for i in range(11)]
        em_label_cat = [self.em_label_cat[i].compute() for i in range(11)]
        f1_label_cat = [self.f1_label_cat[i].compute() for i in range(11)]

        gt_prob = self.gt_prob.compute()
        perplexity = self.perplexity.compute()
        
        return em, f1_score, em_label, f1_score_label, em_cat, f1_cat, em_label_cat, f1_label_cat, gt_prob, perplexity

