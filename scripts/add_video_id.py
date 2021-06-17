import pandas as pd
from lqam.util.file_utils import cached_path
import json
from lqam.methods.dataset import URL_DATA_VAL

URL_RAW_DATA = "https://drive.google.com/uc?id=1rglAizpxanVDejDP6HuCM7UaXLxEQwnK&export=download"
val_data = "https://drive.google.com/uc?id=1si1SNu8ARt8kuAu9ejWFNEAzGf_rtS1m&export=download"

with open(cached_path(URL_RAW_DATA)) as f:
    raw_data = json.load(f)

val = pd.read_csv(cached_path(val_data))

val['video_id'] = ''

vid = {}

for data in raw_data:
    vid[data["masked_caption"]] = data["video_id"]

# for idx, row in val.iterrows():
#     val['video_id'][idx] = vid[row['masked_caption']]

# val = val.rename(columns={'ground_truth':'label'})
# val[['video_id', 'masked_caption', 'label', 'category']].to_csv("val_label_categories.tsv", sep="\t", index=False)

with open(cached_path(URL_DATA_VAL)) as f:
    curr_val = json.load(f)

for data in curr_val:
    if data["masked_caption"] not in vid:
        print(data["masked_caption"])

curr_vid = [data['masked_caption'] for data in curr_val]
for data in raw_data:
    if data['masked_caption'] not in curr_vid:
        print()
        print(data['masked_caption'])