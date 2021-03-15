from typing import Iterable, Iterator, Optional, Sequence

import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.nn.utils.rnn import pad_sequence

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
