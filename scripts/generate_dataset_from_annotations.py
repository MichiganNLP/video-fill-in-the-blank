#!/usr/bin/env python
import argparse
import json

from tqdm.auto import tqdm

from lqam.annotations.metrics import compute_answer_level_metrics
from lqam.annotations.postprocessing import hits_to_instances, parse_hits, unavailable_video_answer
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("annotation_results_path", metavar="ANNOTATION_RESULTS_FILE_OR_URL", type=cached_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    instances = hits_to_instances(parse_hits(args.annotation_results_path))

    formatted_instances = []
    for instance in tqdm(instances.values()):
        answer_level_metrics = compute_answer_level_metrics(instance["question"], instance["answers_by_worker"],
                                                            instance["label"])

        np_answers = [[answer for answer, metrics in worker_answers.items() if metrics["np"]]
                      for worker_id, worker_answers in answer_level_metrics.items() if worker_id != "std_answer"]
        np_answers = [worker_answers for worker_answers in np_answers if worker_answers]

        if not any(unavailable_video_answer(answer) for worker_answers in np_answers for answer in worker_answers):
            formatted_instances.append({
                "video_id": instance["video_id"],
                "video_start_time": instance["video_start_time"],
                "video_end_time": instance["video_end_time"],
                "caption": instance["question"].replace("_____", instance["label"]),
                "masked_caption": instance["question"],
                "label": instance["label"],
                "additional_answers": np_answers,
            })

    print(json.dumps(formatted_instances))


if __name__ == "__main__":
    main()
