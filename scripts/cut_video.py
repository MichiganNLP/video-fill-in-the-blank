#!/usr/bin/env python
# coding: utf-8
import os
import time
from collections import defaultdict

import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def main():
    origin_data = pd.read_csv("input.csv")

    video_dict = defaultdict(list)  # {vid:[[start_time,end_time],[start_time,end_time],[]...], vid:[]}
    for index, row in origin_data.iterrows():
        cid = row["video_id"]
        video_dict[cid].append([int(row["video_start_time"]), int(row["video_end_time"])])

    videos_left = list(video_dict.keys())
    total_left_count = len(videos_left)
    cut_count = 0
    while True:  # keep listening the download video repository
        current_prefixes = [f.split(".")[0] for f in os.listdir("videos")]
        current_suffices = [f.split(".")[1] for f in os.listdir("videos")]

        if "invalid_vid" in current_prefixes:  # remove invalid videos from videos left list
            try:
                with open("videos/invalid_vid.txt") as file:
                    invalid_videos = set(id_[:-1] for id_ in file.readlines())
            except Exception as e:  # "invalid_vid.txt" may be still being written
                print(f"**** {e}\n")
                time.sleep(0.3)
            videos_left = [v for v in videos_left if v not in invalid_videos]

        cut_videos = []
        for video in videos_left:
            if video in current_prefixes:
                try:
                    suffix = "." + current_suffices[current_prefixes.index(video)]  # according suffix
                    name = video + suffix
                    for start_end_time in video_dict[video]:  # one video may related to multiple questions
                        start_time, end_time = start_end_time
                        ffmpeg_extract_subclip(f"videos/{name}", start_time, end_time,
                                               targetname=f"videos/{video}_{start_time}_{end_time}{suffix}")
                    cut_videos.append(video)
                except Exception as e:  # video may be still being written
                    print(f"**** {e}\n")
                    time.sleep(0.3)
        for v in cut_videos:
            videos_left.remove(v)
        if total_left_count - len(videos_left) > cut_count:
            cut_count = total_left_count - len(videos_left)
            print(f"{cut_count}/{total_left_count}")
        if len(videos_left) == 0:  # all videos cut
            break


if __name__ == "__main__":
    main()
