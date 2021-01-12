#!/usr/bin/env python
import argparse
import json
import random
from typing import Any, Iterable, Mapping

import pandas as pd
from tqdm.auto import tqdm

from lqam.core.noun_phrases import create_spacy_model_for_noun_phrase_check
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path
from lqam.util.open_utils import smart_open


def _preprocess_caption(caption: str) -> str:
    caption = caption.strip()
    caption = caption[0].upper() + caption[1:]

    if not caption.endswith("."):
        caption += "."

    return caption


def generate_data(instances: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    spacy_model = create_spacy_model_for_noun_phrase_check()

    selected_data = []

    for instance in tqdm(instances):
        # Just get the first acceptable one, if any.
        for caption in instance["enCap"]:
            caption = _preprocess_caption(caption)

            spacy_doc = spacy_model(caption)

            if not (noun_chunks := list(spacy_doc.noun_chunks)):
                continue

            noun_chunk = random.choice(noun_chunks)

            chunk_start_in_caption = spacy_doc[noun_chunk.start].idx
            chunk_end_in_caption = spacy_doc[noun_chunk.end - 1].idx + len(spacy_doc[noun_chunk.end - 1])

            video_id, video_start_time, video_end_time = instance["videoID"].rsplit("_", maxsplit=2)

            selected_data.append({
                "video_id": video_id,
                "video_start_time": int(video_start_time),
                "video_end_time": int(video_end_time),
                "caption": caption,
                "masked_caption": caption[:chunk_start_in_caption] + "_____" + caption[chunk_end_in_caption:],
                "label": noun_chunk.text,
            })

            break

    return pd.DataFrame(selected_data)


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()

    parser.add_argument("vatex_json_path_or_url", metavar="VATEX_JSON_FILE_OR_URL", nargs="?", default="-")
    parser.add_argument("--seed", type=int, default=2)

    args = parser.parse_args()

    args.input = args.vatex_json_path_or_url if args.vatex_json_path_or_url == "-" \
        else cached_path(args.vatex_json_path_or_url)

    return args


def main() -> None:
    args = parse_args()

    random.seed(args.seed)

    with smart_open(args.input) as file:
        instances = json.load(file)

    df = generate_data(instances)

    print(df.to_csv(index=False))


if __name__ == "__main__":
    main()
