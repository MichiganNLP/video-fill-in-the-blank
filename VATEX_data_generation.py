import random
import math
from typing import Sequence

import torch
import spacy
import pandas as pd
import json

with open("vatex_training_v1.0.json") as file:
    instances_train = json.load(file)

with open("vatex_public_test_english_v1.1.json") as file:
    instances_test = json.load(file)

with open("vatex_validation_v1.0.json") as file:
    instances_val = json.load(file)

nlp_spacy = spacy.load("en_core_web_sm")

# make sure we generate the same data
random.seed(2)

def instance_to_caption_list(instance: Dict[str, Any]) -> Sequence[str]:
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

            chunk = None
            for caption in caption_list:
                caption = preprocess_caption(caption)
                spacy_doc = nlp_spacy(caption)
                if len(list(spacy_doc.noun_chunks)) == 0:
                    continue
                chunk = random.choice(list(spacy_doc.noun_chunks))
                chunk_start_in_caption = spacy_doc[chunk.start].idx
                chunk_end_in_caption = spacy_doc[chunk.end - 1].idx + len(spacy_doc[chunk.end - 1])

                masked_caption = caption[:chunk_start_in_caption] + "<extra_id_0>" + caption[chunk_end_in_caption:]
                ground_truth_label = "<extra_id_0> " + chunk.text + " <extra_id_1>" 
                
                random_choice.append((instance["videoID"], caption, masked_caption, chunk.text))
                chunk_start_in_caption = spacy_doc[chunk.start].idx
                chunk_end_in_caption = spacy_doc[chunk.end - 1].idx + len(spacy_doc[chunk.end - 1])

                masked_caption = caption[:chunk_start_in_caption] + "<extra_id_0>" + caption[chunk_end_in_caption:]
                ground_truth_label = "<extra_id_0> " + chunk.text + " <extra_id_1>"

            if chunk == None:
                continue

            chunk_start_in_caption = spacy_doc[chunk.start].idx
            chunk_end_in_caption = spacy_doc[chunk.end - 1].idx + len(spacy_doc[chunk.end - 1])

            masked_caption = caption[:chunk_start_in_caption] + "<extra_id_0>" + caption[chunk_end_in_caption:]
            ground_truth_label = "<extra_id_0> " + chunk.text + " <extra_id_1>" 

    random.shuffle(random_choice)


    random_df = pd.DataFrame(random_choice, columns=["videoID", "caption", "masked caption", "label"])
    return random_df

folder = "/content/drive/My Drive/"

random_df_train = generate_data(instances_train)
random_df_test = generate_data(instances_test)
random_df_val = generate_data(instances_val)

random_df_train.to_csv(f'{folder}train.csv',index=False)
random_df_val.to_csv(f'{folder}val.csv',index=False)
random_df_test.to_csv(f'{folder}test.csv',index=False)