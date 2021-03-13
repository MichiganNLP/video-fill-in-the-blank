#!/usr/bin/env python
import argparse
import sys
from collections import defaultdict

from lqam.annotations.postprocessing import hits_to_instances, parse_hits
from lqam.methods.dataset import QGenDataModule
from lqam.methods.metrics import ExactMatchAccuracyMany, F1ScoreMany
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("annotation_results_path_or_url", metavar="ANNOTATION_RESULTS_FILE_OR_URL", nargs="?",
                        default="-")

    args = parser.parse_args()

    args.input = sys.stdin if args.annotation_results_path_or_url == "-" \
        else cached_path(args.annotation_results_path_or_url)

    return args


def main() -> None:
    args = parse_args()

    data_module = QGenDataModule(batch_size=None)
    data_loader = data_module.val_dataloader()

    dataset_instance_map = {(instance["video_id"], instance["video_start_time"], instance["video_end_time"],
                             instance["masked_caption"]): instance
                            for instance in data_loader}

    accuracy_by_worker = defaultdict(lambda: ExactMatchAccuracyMany())
    accuracy_many_by_worker = defaultdict(lambda: ExactMatchAccuracyMany())
    f1_score_by_worker = defaultdict(lambda: F1ScoreMany())
    f1_score_many_by_worker = defaultdict(lambda: F1ScoreMany())

    hits = parse_hits(args.input)

    for instance in hits_to_instances(hits).values():
        key = instance["video_id"], instance["video_start_time"], instance["video_end_time"], instance["question"]
        if instance_in_val := dataset_instance_map.get(key):  # FIXME: there's 1 missing in val.
            label = instance_in_val["label"]
            additional_answers = instance_in_val["additional_answers"]

            for worker_id, worker_answers in instance["answers_by_worker"].items():
                first_worker_answer = worker_answers[0]
                accuracy_by_worker[worker_id]([first_worker_answer], [label])
                accuracy_many_by_worker[worker_id]([first_worker_answer], [label], [additional_answers])
                f1_score_by_worker[worker_id]([first_worker_answer], [label])
                f1_score_many_by_worker[worker_id]([first_worker_answer], [label], [additional_answers])

    worker_count = len(accuracy_by_worker)

    print(f"EM label: {sum(m.compute().item() for m in accuracy_by_worker.values()) / worker_count * 100:.1f}%")
    print(f"F1 label: {sum(m.compute().item() for m in accuracy_many_by_worker.values()) / worker_count * 100:.1f}%")
    print(f"EM: {sum(m.compute().item() for m in f1_score_by_worker.values()) / worker_count * 100:.1f}%")
    print(f"F1: {sum(m.compute().item() for m in f1_score_many_by_worker.values()) / worker_count * 100:.1f}%")


if __name__ == "__main__":
    main()
