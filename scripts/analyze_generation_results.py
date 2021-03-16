import json
import pandas as pd

import spacy.tokens

from lqam.methods.dataset import URL_DATA_VAL
from lqam.util.file_utils import cached_path

from lqam.methods.metrics import AlmostExactMatchAccuracy
from lqam.methods.metrics import F1Scores

URL_TEXT_ONLY_PRED = "https://drive.google.com/uc?id=1si1SNu8ARt8kuAu9ejWFNEAzGf_rtS1m&export=download"
URL_MULTI_MODAL_PRED = "https://drive.google.com/uc?id=1Eebdqo7Y7IssDBp12TUm53hCTFMiBNdI&export=download"
URL_RAW_DATA = "https://drive.google.com/uc?id=1rglAizpxanVDejDP6HuCM7UaXLxEQwnK&export=download"

def post_process(raw_data_path, text_only_path, multi_modal_path):
    with open(cached_path(raw_data_path)) as f:
        raw_data = json.load(f)

    text_only = pd.read_csv(cached_path(text_only_path))
    multi_modal = pd.read_csv(cached_path(multi_modal_path))

    output = {}

    # prediction files don't contain video ids, so we use masked captions as keys
    for data in raw_data:
        output[data["masked_caption"]] = {"label": data["label"], "additional_answers": data["additional_answers"]}
    
    for idx, row in text_only.iterrows():
        output[row["masked_caption"]]["text_only_pred"] = row["generated"]
        output[row["masked_caption"]]["category"] = row["category"]

    for idx, row in multi_modal.iterrows():
        output[row["masked_caption"]]["multi_modal_pred"] = row["generated"]

    return output

def compute_freq(df):
    return df["generated"].value_counts()

# compute noun frequency in answers
def compute_freq_noun(df):
    noun_freq = {}

predictions = post_process(URL_RAW_DATA, URL_TEXT_ONLY_PRED, URL_MULTI_MODAL_PRED)
text_only_em_list = [AlmostExactMatchAccuracy() for i in range(11)]
text_only_f1_list = [F1Scores() for i in range(11)]
multi_modal_em_list = [AlmostExactMatchAccuracy() for i in range(11)]
multi_modal_f1_list = [F1Scores() for i in range(11)]

text_only_em_without_aa_list = [AlmostExactMatchAccuracy() for i in range(11)]
text_only_f1_without_aa_list = [F1Scores() for i in range(11)]
multi_modal_em_without_aa_list = [AlmostExactMatchAccuracy() for i in range(11)]
multi_modal_f1_without_aa_list = [F1Scores() for i in range(11)]

category_num = [0] * 11

# diff1 & diff2: with additional answers
# diff1: mm modal predicts correctly but text-only model doesn't
diff1 = []
# diff2: text-only model predicts correctly but mm modal doesn't
diff2 = []

# diff3 & diff4: without additional answers
# diff3: mm modal predicts correctly but text-only model doesn't
diff3 = []
# diff4: text-only model predicts correctly but mm modal doesn't
diff4 = []

total = len(predictions)
# avg pred len
avg_pred_len_text_only = 0
avg_pred_len_multi_modal = 0
avg_label_len = 0

total_pred_len_text_only_category = [0]*11
total_pred_len_multi_modal_category = [0]*11
total_label_len_category = [0] * 11

for masked_caption in predictions:
    pred = predictions[masked_caption]
    category = pred["category"]
    label = pred["label"]
    additional_answers = pred["additional_answers"]
    
    category_num[category] += 1

    text_only_pred = pred["text_only_pred"]
    multi_modal_pred = pred["multi_modal_pred"]

    text_only_pred_len = len(text_only_pred.split())
    multi_modal_pred_len = len(multi_modal_pred.split())
    label_len = len(label.split())
    avg_pred_len_text_only += text_only_pred_len
    total_pred_len_text_only_category[category] += text_only_pred_len
    avg_pred_len_multi_modal += multi_modal_pred_len
    total_pred_len_multi_modal_category[category] += multi_modal_pred_len
    avg_label_len += label_len
    total_label_len_category[category] += label_len
    
    text_only_score = text_only_em_list[category]([text_only_pred], [label], [additional_answers])
    text_only_f1_list[category]([text_only_pred], [label], [additional_answers])
    multi_modal_score = multi_modal_em_list[category]([multi_modal_pred], [label], [additional_answers])
    multi_modal_f1_list[category]([multi_modal_pred], [label], [additional_answers])

    if text_only_score.item() == 0 and multi_modal_score.item() == 1:
        diff1.append((masked_caption, label, text_only_pred, multi_modal_pred))

    if text_only_score.item() == 1 and multi_modal_score.item() == 0:
        diff2.append((masked_caption, label, text_only_pred, multi_modal_pred)) 

    text_only_score_without_aa = text_only_em_without_aa_list[category]([text_only_pred], [label])
    text_only_f1_without_aa_list[category]([text_only_pred], [label])
    multi_modal_score_without_aa = multi_modal_em_without_aa_list[category]([multi_modal_pred], [label])
    multi_modal_f1_without_aa_list[category]([multi_modal_pred], [label])

    if text_only_score_without_aa.item() == 0 and multi_modal_score_without_aa.item() == 1:
        diff3.append((masked_caption, label, text_only_pred, multi_modal_pred))

    if text_only_score_without_aa.item() == 1 and multi_modal_score_without_aa.item() == 0:
        diff4.append((masked_caption, label, text_only_pred, multi_modal_pred)) 

text_only_em = [text_only_em_list[i].compute().item() for i in range(11)]
text_only_f1 = [text_only_f1_list[i].compute().item() for i in range(11)]
multi_modal_em = [multi_modal_em_list[i].compute().item() for i in range(11)]
multi_modal_f1 = [multi_modal_f1_list[i].compute().item() for i in range(11)]

text_only_em_without_aa = [text_only_em_without_aa_list[i].compute().item() for i in range(11)]
text_only_f1_without_aa = [text_only_f1_without_aa_list[i].compute().item() for i in range(11)]
multi_modal_em_without_aa = [multi_modal_em_without_aa_list[i].compute().item() for i in range(11)]
multi_modal_f1_without_aa = [multi_modal_f1_without_aa_list[i].compute().item() for i in range(11)]

stats = pd.DataFrame(data={
                            "text_only_em": text_only_em,
                            "text_only_em_without_aa": text_only_em_without_aa,
                            "multi_modal_em": multi_modal_em,
                            "multi_modal_em_without_aa": multi_modal_em_without_aa,
                            "text_only_f1": text_only_f1,
                            "text_only_f1_without_aa": text_only_f1_without_aa,                            
                            "multi_modal_f1": multi_modal_f1,
                            "multi_modal_f1_without_aa": multi_modal_f1_without_aa
                        })
print(category_num)
print(stats)

df_diff1 = pd.DataFrame(diff1, columns=["masked_caption", "label", "text-only pred", "multi-modal pred"])

avg_pred_len_text_only /= total
avg_pred_len_multi_modal /= total
avg_label_len /= total
avg_pred_len_text_only_category = [total_pred_len_text_only_category[i]/category_num[i] for i in range(11)]
avg_pred_len_multi_modal_category = [total_pred_len_multi_modal_category[i]/category_num[i] for i in range(11)]
avg_label_len_category = [total_label_len_category[i]/category_num[i] for i in range(11)]

print(avg_pred_len_text_only)
print(avg_pred_len_multi_modal)
print(avg_label_len)
print(avg_pred_len_text_only_category)
print(avg_pred_len_multi_modal_category)
print(avg_label_len_category)

pass