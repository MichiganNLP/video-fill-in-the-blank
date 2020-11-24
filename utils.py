from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from math import floor
import numpy as np
import torch
import json
import pickle
import h5py
import os
import math


def fit(train_text_file, num_tokens):
    # word_to_idx_file = f'
    # /scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/latest_data/word_to_idx-{num_tokens}.pkl'
    stop_words = set(stopwords.words('english'))
    word_to_idx, idx_to_word = dict(), dict()
    # collect all documents into all_sentences
    documents = list()
    with open(train_text_file, 'r') as f:
        data = json.load(f)
        for key in data:
            documents.extend(data[key]['sentences'])
    # tokenize documents
    tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
    # filter out stop words
    tokens = [w for t_doc in tokenized_docs for w in t_doc if w.isalpha()]
    tokens_no_stops = [t for t in tokens if t not in stop_words]
    # Get top k tokens by frequency
    counter = Counter(tokens_no_stops)
    top_k_words = counter.most_common(num_tokens)
    # create (word, index) dictionary
    for word, _ in top_k_words:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
            idx_to_word[len(idx_to_word)] = word
    # open pickle file and dump data into the file
    # pickle_file = open(word_to_idx_file, 'wb')
    # pickle.dump((word_to_idx, len(tokens)), pickle_file)
    # pickle_file.close()
    return idx_to_word, word_to_idx, len(word_to_idx)


def make_bow_vector(sentence, word_to_idx):
    stop_words = set(stopwords.words('english'))
    vector = torch.zeros(len(word_to_idx), dtype=torch.float32)
    tokens = [w for w in word_tokenize(sentence) if w.isalpha()]
    token_no_stops = [t for t in tokens if t not in stop_words]
    for word in token_no_stops:
        if word in word_to_idx:
            vector[word_to_idx[word]] += 1
    return vector


def make_target(label, label_to_ix):
    if label_to_ix.get(label):
        return torch.LongTensor([label_to_ix[label]])
    else:
        return torch.LongTensor([0])


def get_video_representations(key, start, end, video_features, video_duration):
    video_feature_len = video_features[key]['c3d_features'].shape[0]

    duration = float(video_duration[key])

    start_frame = math.floor(start / duration * video_feature_len)
    end_frame = math.floor(end / duration * video_feature_len)

    feature_np = video_features[key]['c3d_features'][start_frame:end_frame + 1]

    if feature_np.shape[0] > 200:
        feature = np.zeros((200, feature_np.shape[1]))
        for i in range(200):
            feature[i] = feature_np[round(i * (feature_np.shape[0] - 1) / 199)]
    else:
        feature = feature_np

    return torch.tensor(feature, dtype=torch.float32)


def build_representation(masked_data_file, video_feature_file,
                         bow_representation_file, video_duration,
                         word_to_idx, num_vocabs):
    # check if representation file exists
    if os.path.exists(bow_representation_file):
        print(f'{bow_representation_file}: representations are loaded.')
        return
    # construct word to index dictionary for sentence representation using training data file: train.json, etc
    # word_to_idx, num_vocabs = fit(train_text_file, num_tokens)
    # retrieve video features from file
    video_features = h5py.File(video_feature_file, 'r')
    # data list will be saved into pickle file - each element is an tuple:
    # (sentence_representation, feature representation, label_representation)
    data = list()
    with open(masked_data_file, 'r') as train_file:
        error_file = open(masked_data_file + '_error_time_stamps.txt', 'w')
        while True:
            key = train_file.readline().strip()
            # check if it is the end of train file
            if not key:
                break
            raw_sentence = train_file.readline().strip()
            sentence = raw_sentence.replace('[MASK]', '')
            label = train_file.readline().strip()
            tt_start, tt_end = json.loads(train_file.readline().strip())
            tt_start, tt_end = float(tt_start), float(tt_end)
            # escape the new line separator
            train_file.readline()
            # create sentence feature and video feature
            sentence_feature = make_bow_vector(sentence, word_to_idx)
            video_feature = get_video_representations(key, tt_start, tt_end,
                                                      video_features, video_duration)
            if video_feature.shape[0] == 0:
                error_file.write('shape: ' + str(video_feature.shape) + '\n')
                error_file.write('key: ' + key + '\n')
                error_file.write('sentence: ' + sentence + '\n')
                error_file.write('label: ' + label + '\n')
                error_file.write('indices: [' + str(tt_start) + ' ' + str(tt_end) + ']' + '\n')
                error_file.write('\n')
            else:
                # average on time dimension
                video_feature = torch.mean(video_feature, dim=0)
                label_representation = make_target(label, word_to_idx)
                data.append((sentence_feature, video_feature, label_representation, key, raw_sentence))
        error_file.close()
        # open pickle file and dump data into the file
        pickle_file = open(bow_representation_file, 'wb')
        pickle.dump(data, pickle_file)
        pickle_file.close()
