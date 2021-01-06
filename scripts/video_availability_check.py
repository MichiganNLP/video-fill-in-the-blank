from pyyoutube import Api
import os
import csv

api = Api(api_key="AIzaSyBC-MJgNI9Wjqz1bOhgjr9LNP3w7hSbFmY")

file_folder = "/home/ruoyao/Downloads/vatex_qa/text_data_20201223/"

for file_name in os.listdir(file_folder):
    name = file_name[:-4]
    with open(os.path.join(file_folder,file_name), 'r') as csvfile:
        unavailable = []
        reader = csv.reader(csvfile, delimiter=',')
        isHead = True
        for row in reader:
            if isHead:
                isHead = False
                continue
            video_id = row[0][:-14]
            video = api.get_video_by_id(video_id=video_id)
            if len(video.items) == 0:
                unavailable.append(video_id)
    with open(name, 'w') as f:
        for v in unavailable:
            f.write(v)
            f.write('\n')
