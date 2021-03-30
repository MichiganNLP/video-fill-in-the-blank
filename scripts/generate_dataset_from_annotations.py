#!/usr/bin/env python
import argparse
import json

from lqam.annotations.postprocessing import filter_and_process_annotated_instances, hits_to_instances, parse_hits
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("annotation_results_path", metavar="ANNOTATION_RESULTS_FILE_OR_URL", type=cached_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    instances = hits_to_instances(parse_hits(args.annotation_results_path)).values()

    print(json.dumps([
        {
            "video_id": instance["video_id"],
            "video_start_time": instance["video_start_time"],
            "video_end_time": instance["video_end_time"],
            "caption": instance["question"].replace("_____", instance["label"]),
            "masked_caption": instance["question"],
            "label": instance["label"],
            "additional_answers": instance["np_answers"],
        } for instance in filter_and_process_annotated_instances(instances)
    ]))


if __name__ == "__main__":
    main()
