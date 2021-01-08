#!/usr/bin/env python
# coding: utf-8
import fileinput

import youtube_dl
from tqdm import tqdm


def main():
    for video_id in tqdm(fileinput.input()):
        try:
            with youtube_dl.YoutubeDL({"format": "best[ext=mp4]/best", "outtmpl": "videos/%(id)s.%(ext)s"}) as dl:
                dl.download([f"https://www.youtube.com/embed/{video_id}"])
        except Exception:  # TODO: make this exception more specific.
            print(video_id)


if __name__ == "__main__":
    main()
