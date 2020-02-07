from torch.utils.data import Dataset
from nltk.corpus import wordnet
from transformers import BertTokenizer
import json
import math
import nltk


class ActivityNetCaptionDataset(Dataset):

    def __init__(self, data_file, word_dict):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = self.parseData(data_file)
        self.word_dict = word_dict
    
    def gen(self, text, parsed_sentence):
        position = []
        for i in range(len(parsed_sentence)):
            if parsed_sentence[i][1] == 'JJ' or parsed_sentence[i][1] == 'NN':
                position.append(i)

        while len(position): 
            idx = random.sample(position, 1)[0]
            if text[idx] in word_dict and word_dict[text[idx]] >= THRESHOLD:
                position.remove(idx)
                continue
            new_sentence = text[:]
            correct_word = new_sentence[idx]
            new_sentence[idx] = '[MASK]'
            return new_sentence, correct_word

    def parseData(self, data_file):
        with open(data_file, 'r') as f:
            raw = json.load(f)
        
        data = []
        for key in raw.keys():
            total_events = len(raw[key]['sentences'])
            for i in range(total_events):
                start_frame = math.floor(raw[key]['timestamps'][i][0] * 2)
                end_frame = math.ceil(raw[key]['timestamps'][i][1] * 2)
                sentence = raw[key]['sentences'][i]
                text = nltk.word_tokenize(sentence.strip().lower())
                parsed_sentence = nltk.pos_tag(text)
                masked_sentence, label = self.gen(text, parsed_sentence)
                data.append((key, start_frame, end_frame, masked_sentence, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 