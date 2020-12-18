import random
import math
from typing import Sequence
from tqdm.auto import tqdm

import torch
import spacy
import pandas as pd
import json

folder = "/scratch/mihalcea_root/mihalcea1/shared_data/qgen/VATEX/multimodal_model/VATEX/"

with open(f"{folder}vatex_training_v1.0.json") as file:
    instances_train = json.load(file)

with open(f"{folder}vatex_public_test_english_v1.1.json") as file:
    instances_test = json.load(file)

with open(f"{folder}vatex_validation_v1.0.json") as file:
    instances_val = json.load(file)

nlp_spacy = spacy.load("en_core_web_sm")

# make sure we generate the same data
random.seed(2)

def instance_to_caption_list(instance):
    return instance["enCap"]  # Just pick the first one.

def preprocess_caption(caption: str) -> str:
    caption = caption.strip()
    caption = caption[0].upper() + caption[1:]

    if not caption.endswith("."):
        caption += "."

    return caption

def generate_data(instances):
    random_choice = []
    with torch.no_grad():
        for instance in tqdm(instances[:10]):
            caption_list = instance_to_caption_list(instance)

            for caption in caption_list:
                caption = preprocess_caption(caption)
                spacy_doc = nlp_spacy(caption)
                if len(list(spacy_doc.noun_chunks)) == 0:
                    continue
                chunk = random.choice(list(spacy_doc.noun_chunks))
                chunk_start_in_caption = spacy_doc[chunk.start].idx
                chunk_end_in_caption = spacy_doc[chunk.end - 1].idx + len(spacy_doc[chunk.end - 1])

                masked_caption = caption[:chunk_start_in_caption] + "<extra_id_0>" + caption[chunk_end_in_caption:]
                
                random_choice.append((instance["videoID"], caption, masked_caption, chunk.text))
                break


    random.shuffle(random_choice)


    random_df = pd.DataFrame(random_choice, columns=["videoID", "caption", "masked caption", "label"])
    return random_df


random_df_train = generate_data(instances_train)
random_df_test = generate_data(instances_test)
random_df_val = generate_data(instances_val)

random_df_train.to_csv(f'{folder}train.csv',index=False)
random_df_val.to_csv(f'{folder}val.csv',index=False)
random_df_test.to_csv(f'{folder}test.csv',index=False)