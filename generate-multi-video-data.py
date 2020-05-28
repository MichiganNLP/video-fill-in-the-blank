# coding: utf-8
# This script generates data for Mturk multi video page
# The algorithm prevents the same video from appearing on the same page as possible

import pandas as pd
from math import ceil
from random import shuffle,randint

VideosPerPage = 5


def find_maximum_N(LeftSpace,N): # find the indexes of the maximum N values of LeftSpace
    left_space = LeftSpace[:]
    temp = []
    for i in range(N):
        max_index = left_space.index(max(left_space))
        temp.append(max_index)
        left_space[max_index] -= 1
    return temp


if __name__ == '__main__':
    one_video_data = pd.read_csv("mturk_trail_val1_50.csv")
    NumberOfQuestons = one_video_data.shape[0]

    # prepare boxes
    number_of_box = ceil(NumberOfQuestons/5)  # number of boxes(webpages) needed
    boxes = [[] for _ in range(number_of_box)]     # real boxes
    box_left_space = [5 for _ in range(number_of_box)]

    # allocate questions to boxes
    current_video = None
    current_lines = []  # lines with the same video_id
    for line in range(NumberOfQuestons):
        videa_id = one_video_data.iloc[line,0]
        if videa_id == current_video:
            current_lines.append(line)
        elif current_video is None:
            current_video = videa_id
            current_lines.append(line)
        else:          # next video_id
            shuffle(current_lines)
            boxes_to_put = find_maximum_N(box_left_space,len(current_lines))
            for ind in range(len(current_lines)):
                boxes[boxes_to_put[ind]].append(current_lines[ind])
                box_left_space[boxes_to_put[ind]] -= 1  
            current_lines = [line]
            current_video = videa_id
            
    if len(current_lines) != 0:     # last video_id
        shuffle(current_lines)
        boxes_to_put = find_maximum_N(box_left_space,len(current_lines))
        for ind in range(len(current_lines)):
            boxes[boxes_to_put[ind]].append(current_lines[ind])


    # boxes to Dataframe
    csv_lines = []
    for box in boxes:
        if len(box)<VideosPerPage:
            box.append(randint(0,NumberOfQuestons-1))
        shuffle(box)
        new_line = []
        for line_num in box:
            new_line += list(one_video_data.iloc[line_num])
        csv_lines.append(new_line)

    # Dataframe to CSV
    titles = ["video1_id","question1","pos_tag1","video1_start_time","video1_end_time",
            "video2_id","question2","pos_tag2","video2_start_time","video2_end_time",
            "video3_id","question3","pos_tag3","video3_start_time","video3_end_time",
            "video4_id","question4","pos_tag4","video4_start_time","video4_end_time",
            "video5_id","question5","pos_tag5","video5_start_time","video5_end_time"]

    df = pd.DataFrame(csv_lines,columns=titles)
    df.to_csv("mturk_val1_50_5videos.csv",encoding="utf8",index=False)
