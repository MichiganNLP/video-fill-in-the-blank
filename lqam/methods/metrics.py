from __future__ import annotations

from collections import Iterable, Iterator, Mapping, MutableMapping, Sequence

import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.nn.utils.rnn import pad_sequence

from lqam.core.metrics import compute_token_level_f1_many, exact_match_many, flatten_all_answers, \
    tokenize_answer_to_compute_metrics
from lqam.methods.dataset import N_CATEGORIES, load_label_categories


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
               additional_answers_batch: Sequence[Sequence[Sequence[str]]] | None = None) -> None:
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
               additional_answers_batch: Sequence[Sequence[Sequence[str]]] | None = None) -> None:  # noqa
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


class AllMetrics:
    def __init__(self, compute_prob: bool = True) -> None:
        self.label_categories = load_label_categories()
        self.metrics: MutableMapping[str, pl.metrics.Metric | Iterable[pl.metrics.Metric]] = {
            "accuracy": ExactMatchAccuracyMany(),
            "f1_score": F1ScoreMany(),
            "accuracy_label": ExactMatchAccuracyMany(),
            "f1_score_label": F1ScoreMany(),
            "accuracy_cat": [ExactMatchAccuracyMany() for _ in range(N_CATEGORIES)],
            "f1_score_cat": [F1ScoreMany() for _ in range(N_CATEGORIES)],
            "accuracy_label_cat": [ExactMatchAccuracyMany() for _ in range(N_CATEGORIES)],
            "f1_score_label_cat": [F1ScoreMany() for _ in range(N_CATEGORIES)],
        }

        if compute_prob:
            self.metrics["ground_truth_prob"] = Average()
            self.metrics["perplexity"] = Perplexity()

    def reset(self):
        for metrics in self.metrics.values():
            if isinstance(metrics, pl.metrics.Metric):
                metrics.reset()
            else:
                for metric in metrics:
                    metric.reset()

    def __call__(self, video_ids: Sequence[str], labels: Sequence[str],
                 additional_answers_batch: Sequence[Sequence[Sequence[str]]], preds: Sequence[str],
                 label_prob: torch.Tensor | None = None, label_probs: torch.Tensor | None = None,
                 perplexity_mask: torch.Tensor | None = None) -> Mapping[str, torch.Tensor]:
        output = {
            "accuracy": self.metrics["accuracy"](preds, labels, additional_answers_batch),
            "f1_score": self.metrics["f1_score"](preds, labels, additional_answers_batch),
            "accuracy_label": self.metrics["accuracy_label"](preds, labels),
            "f1_score_label": self.metrics["f1_score_label"](preds, labels),
        }

        for video_id, label, pred, additional_answers in zip(video_ids, labels, preds, additional_answers_batch):
            cat = self.label_categories.get(video_id, N_CATEGORIES - 1)
            output[f"accuracy_cat_{cat}"] = self.metrics["accuracy_cat"][cat]([pred], [label], [additional_answers])
            output[f"f1_score_cat_{cat}"] = self.metrics["f1_score_cat"][cat]([pred], [label], [additional_answers])
            output[f"accuracy_label_cat_{cat}"] = self.metrics["accuracy_label_cat"][cat]([pred], [label])
            output[f"f1_score_label_cat_{cat}"] = self.metrics["f1_score_label_cat"][cat]([pred], [label])

        if ground_truth_prob_metric := self.metrics.get("ground_truth_prob"):
            assert label_prob is not None
            output["ground_truth_prob"] = ground_truth_prob_metric(label_prob)

        if perplexity_metric := self.metrics.get("perplexity"):
            assert label_probs is not None and perplexity_mask is not None
            output["perplexity"] = perplexity_metric(label_probs, perplexity_mask)

        return output

    def compute(self) -> Mapping[str, torch.Tensor]:
        output = {}
        for name, metrics in self.metrics.items():
            if isinstance(metrics, pl.metrics.Metric):
                output[name] = metrics.compute()
            else:
                output.update({f"{name}_{i}": metric.compute() for i, metric in enumerate(metrics)})
        return output
