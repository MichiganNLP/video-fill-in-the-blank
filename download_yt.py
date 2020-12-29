#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import youtube_dl


def download_videos(vids):
    numovs = len(vids)
    invalid_vids = []
    for i in range(numovs):
        vid = vids[i]
        try:  # video is valid
            videoURL = 'https://www.youtube.com/embed/' + vid
            videoSavePath = './tmpt-video/'
            ydl_opts = {
                # 'proxy': 'socks5://127.0.0.1:1234',
                'format': 'best[ext=mp4]/best',  # saving format
                'outtmpl': videoSavePath + "%(id)s.%(ext)s",  # default as '%(title)s-%(id)s.%(ext)s'
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:  # download video
                ydl.download([videoURL])
        except Exception as ex:  # video may is invalid
            print("**** %s\n" % ex)
            invalid_vids.append(vid + "\n")
        print("%d/%d" % (i + 1, numovs))
    with open("tmpt-video/invalid_vid.txt", "w") as f:  # output invalid video ids
        f.writelines(invalid_vids)
    print("Download Finished!!!")


if __name__ == "__main__":
    origin_data = pd.read_csv("input.csv")
    videos = list(set(origin_data["video_id"]))
    download_videos(videos)
