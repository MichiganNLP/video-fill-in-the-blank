import torch
import torch.nn as nn
import h5py
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter


folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions"
videoFeatures = h5py.File(f"{folder}/ActivityNet_Captions_Video_Features/sub_activitynet_v1-3.c3d.hdf5", 'r')
trainTextFile = f"{folder}/train.json"
valTextFile = f"{folder}/val_1.json"

stop_words = set(stopwords.words('english'))


class baseline_BOW_VF(nn.Module):
    def __init__(self, W_D_in, V_D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(baseline_BOW_VF, self).__init__()
        self.linear1 = nn.Linear(W_D_in, V_D_in)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(2 * V_D_in, D_out)

    def forward(self, text_feature, video_feature):
        text_feature_out = self.relu1(self.linear1(text_feature))
        fused_feature = torch.cat([text_feature_out, video_feature], dim=1)
        y_pred = self.linear2(fused_feature)
        return y_pred


def fit(train_text_file, num_tokens):
    """
        train_text_file:
    """
    word_to_idx = dict()
    file = open(train_text_file, 'r')
    data = json.load(file)
    # collect all documents into all_sentences
    documents = list()
    for key in data:
        documents.extend(data[key]['sentences'])
    # tokenize documents
    tokenzied_docs = [word_tokenize(doc.lower()) for doc in documents]
    # filter out stop words
    tokens = [w for t_doc in tokenzied_docs for w in t_doc if w.isalpha()]
    tokens_no_stops = [t for t in tokens if t not in stop_words]
    # Get top k tokens by frequency
    top_k_words = Counter(tokens_no_stops).most_common(num_tokens)
    # create (word, index) dictionary
    for word, _ in top_k_words:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
    return word_to_idx


def transform(sentence, word_to_idx):
    vector = torch.zeros(len(word_to_idx))
    tokens = [w for w in word_tokenize(sentence) if w.isalpha()]
    token_no_stops = [t for t in tokens if t not in stop_words]
    for word in token_no_stops:
        if word in word_to_idx:
            vector[word_to_idx[word]] += 1
    return vector.view(1, -1)