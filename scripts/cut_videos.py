#!/usr/bin/env python
# coding: utf-8
import fileinput
import os
import time
from collections import defaultdict

import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def main():
    df = pd.read_csv(fileinput.input())

    video_dict = defaultdict(list)
    for row in df.itertuples():
        video_dict[row.video_id].append((row.video_start_time, row.video_end_time))

    videos_left = list(video_dict.keys())
    total_left_count = len(videos_left)
    cut_count = 0
    while videos_left:  # Keep listening the download video repository.
        filenames_without_ext, extensions = zip(*(name.rsplit(".", maxsplit=1) for name in os.listdir("videos")))

        if "invalid_vid" in filenames_without_ext:  # Remove invalid videos from videos left list.
            try:
                with open("videos/invalid_vid.txt") as file:
                    invalid_videos = {id_[:-1] for id_ in file.readlines()}
            except Exception as e:  # It may be still being written.
                print(f"**** {e}\n")
                time.sleep(0.3)

            videos_left = [id_ for id_ in videos_left if id_ not in invalid_videos]

        cut_videos = []
        for id_ in videos_left:
            if id_ in filenames_without_ext:
                ext = extensions[filenames_without_ext.index(id_)]
                name = f"{id_}.{ext}"
                try:
                    for start_time, end_time in video_dict[id_]:
                        ffmpeg_extract_subclip(f"videos/{name}", start_time, end_time,
                                               targetname=f"videos/{id_}_{start_time}_{end_time}.{ext}")
                    cut_videos.append(id_)
                except Exception as e:  # The video may be still being written.
                    print(f"**** {e}\n")
                    time.sleep(0.3)
        for id_ in cut_videos:
            videos_left.remove(id_)
        if total_left_count - len(videos_left) > cut_count:
            cut_count = total_left_count - len(videos_left)
            print(f"{cut_count}/{total_left_count}")


if __name__ == "__main__":
    main()
