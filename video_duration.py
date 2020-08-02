import json
import pickle

folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/20200729before"

train = 'train.json'
val1 = 'val_1.json'
val2 = 'val_2.json'

with open(f"{folder}/{train}", 'r') as f:
    train_data = json.load(f)

with open(f"{folder}/{val1}", 'r') as f:
    val1_data = json.load(f)

with open(f"{folder}/{val2}", 'r') as f:
    val2_data = json.load(f)

duration = {}

for key in train_data:
    duration[key] = train_data[key]["duration"]

for key in val1_data:
    if key in duration and duration[key] != val1_data[key]["duration"]:
        print(key)
        print(duration[key])
        print(val1_data[key]["duration"])
    duration[key] = val1_data[key]["duration"]

for key in val2_data:
    if key in duration and duration[key] != val2_data[key]["duration"]:
        print(key)
        print(duration[key])
        print(val2_data[key]["duration"])
    duration[key] = val2_data[key]["duration"]

with open("video_duration.pkl", 'wb') as f:
    pickle.dump(duration, f)