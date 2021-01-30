#!/usr/bin/env python
import argparse
import random
import sys
from collections import defaultdict

import pandas as pd
import pytorch_lightning as pl
from tqdm.auto import tqdm

from lqam.annotations.metrics import compute_answer_level_annotation_metrics
from lqam.annotations.postprocessing import format_answer, hits_to_instances, parse_hits
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("annotation_results_path_or_url", metavar="ANNOTATION_RESULTS_FILE_OR_URL", nargs="?",
                        default="-")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--min-good-answers-per-question", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    args.annotation_results_path = sys.stdin if args.annotation_results_path_or_url == "-" \
        else cached_path(args.annotation_results_path_or_url)

    return args


def main() -> None:
    args = parse_args()

    pl.seed_everything(args.seed)

    hits = parse_hits(args.annotation_results_path)
    instances = hits_to_instances(hits)

    instances_by_worker_id = defaultdict(list)
    for instance in tqdm(instances.values()):
        instance["answers"] = {worker_id: [format_answer(answer) for answer in answers]
                               for worker_id, answers in instance["answers"].items()}

        answer_level_metrics = compute_answer_level_annotation_metrics(instance["question"], instance["answers"],
                                                                       instance["label"])

        for worker_id, answers in instance["answers"].items():
            instances_by_worker_id[worker_id].append({
                **instance,
                "answers": [answer for answer in answers if answer_level_metrics[worker_id][answer]["np"]],
            })

    for worker_id, worker_instances in list(instances_by_worker_id.items()):
        np_answers_per_question = sum(len(instance["answers"]) for instance in worker_instances) / len(worker_instances)
        if np_answers_per_question < args.min_good_answers_per_question:
            del instances_by_worker_id[worker_id]

    for worker_id in instances_by_worker_id:
        if len(instances_by_worker_id[worker_id]) > args.sample_size:
            instances_by_worker_id[worker_id] = random.sample(instances_by_worker_id[worker_id], args.sample_size)

    df = pd.DataFrame([
        {
            "worker_id": worker_id,
            **{k: v for k, v in instance.items() if k not in {"answers", "np"}},
            "answer": answer,
        }
        for worker_id, worker_instances in instances_by_worker_id.items()
        for instance in worker_instances
        for answer in instance["answers"]
    ])

    print(df.to_csv(index=False))


if __name__ == "__main__":
    main()
