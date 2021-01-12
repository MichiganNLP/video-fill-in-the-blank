#!/usr/bin/env python
import fileinput
import itertools
import os
from typing import Iterable, Iterator

import pyyoutube
from tqdm.auto import tqdm

from lqam.util.iterable_utils import chunks

MAX_VIDEOS_PER_REQUEST = 50


def are_videos_available(api: pyyoutube.Api, video_ids: Iterable[str]) -> Iterator[bool]:
    for video_id_batch in chunks(video_ids, MAX_VIDEOS_PER_REQUEST):
        available_videos = {video.id for video in api.get_video_by_id(video_id=video_id_batch).items}
        for video_id in video_id_batch:
            yield video_id in available_videos


def main():
    api = pyyoutube.Api(api_key=os.environ["GOOGLE_API_KEY"])

    video_id_generator1, video_id_generator2 = itertools.tee(stripped_line
                                                             for line in fileinput.input()
                                                             if (stripped_line := line.strip()))

    for video_id, is_available in tqdm(zip(video_id_generator1, are_videos_available(api, video_id_generator2))):
        if not is_available:
            print(video_id)


if __name__ == "__main__":
    main()
