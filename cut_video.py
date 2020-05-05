# coding: utf-8
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from collections import OrderedDict, defaultdict
import os
import time


if __name__ == "__main__":
    origin_data = pd.read_csv("input.csv")
    # video information dictionary
    video_dict = dict()      # {vid:[[start_time,end_time],[start_time,end_time],[]...], vid:[]}
    for index, row in origin_data.iterrows():
        cid = row['video_id']
        if cid in video_dict:
            video_dict[cid].append([int(row["video_start_time"]),int(row["video_end_time"])])
        else:
            video_dict[cid] = [[int(row["video_start_time"]),int(row["video_end_time"])]]
    
    # Cut videos
    left_vids = list(video_dict.keys()) # videos left to be cut
    total_left_number = len(left_vids)
    cutted_number = 0

    while(True):    # keep listening the download video repository
        current_pres = [f.split(".")[0] for f in os.listdir("tmpt-video")]   # preffix of files, video file may be not .mp4 file
        current_suffs = [f.split(".")[1] for f in os.listdir("tmpt-video")]  # suffix of files

        if "invalid_vid" in current_pres:       # remove invalid videos from videos left list
            try:
                invalid_vids = []
                with open("tmpt-video/invalid_vid.txt","r") as f:
                    invalid_vids = [id[:-1] for id in f.readlines()]
                left_vids = list(filter(lambda x: x not in invalid_vids, left_vids))
            except Exception as ex:             # "invalid_vid.txt" may be still being written
                print("**** %s\n"%ex)
                time.sleep(0.3)
        
        # cut left videos
        cutted_vids = []
        for vid in left_vids:     # left videos
            if vid in current_pres:
                try:
                    vid_suff = "." + current_suffs[current_pres.index(vid)]  # according suffix
                    video_name = vid + vid_suff
                    for setime in video_dict[vid]:  # one video may related to multiple questions
                        ffmpeg_extract_subclip("tmpt-video/"+video_name, setime[0],setime[1],
                                            targetname="video/"+vid+"_"+str(setime[0])+"_"+str(setime[1])+vid_suff)
                    cutted_vids.append(vid)
                except Exception as ex:             # video may be still being written
                    print("**** %s\n"%ex)
                    time.sleep(0.3)
        for v in cutted_vids: left_vids.remove(v)
        if total_left_number-len(left_vids) > cutted_number:
            cutted_number = total_left_number-len(left_vids)
            print("%d/%d"%(cutted_number,total_left_number))
        if len(left_vids) == 0: break               # all videos cutted 
    
    print("Cut Finished!!!")
