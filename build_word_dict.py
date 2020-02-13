import pickle
import json

trainTextFile = "/scratch/mihalcea_root/mihalcea1/ruoyaow/ActivityNet_Captions/train.json"
valTextFile = "/scratch/mihalcea_root/mihalcea1/ruoyaow/ActivityNet_Captions/val_1.json"
word_dict = {}
wordID = 0

with open(trainTextFile, 'r') as f:
    train_data = json.load(f)

with open(valTextFile, 'r') as f:
    val_data = json.load(f)

def count_words(data):
    for key in data:
        total_events = len(data[key]['sentences'])
        for i in range(total_events):
            sentence = data[key]['sentences'][i]
            word_list = sentence.strip().split(' ')
            for word in word_list:
                if word in word_dict:
                    word_dict[word]["freq"] += 1
                else:
                    word_dict[word] = {"id": wordID, "freq": 1}
                    wordID += 1

count_words(train_data)
count_words(val_data)

with open('word_dict.pkl', 'wb') as f:
    pickle.dump(word_dict, f)