import pickle
import json
import nltk

trainTextFile = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/train.json"
valTextFile = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/val_1.json"
word_dict = {}
wordID = 0

with open(trainTextFile, 'r') as f:
    train_data = json.load(f)

with open(valTextFile, 'r') as f:
    val_data = json.load(f)

def count_words(data, startWordID):
    newWordID = startWordID
    for key in data:
        total_events = len(data[key]['sentences'])
        for i in range(total_events):
            sentence = data[key]['sentences'][i]
            word_list = nltk.word_tokenize(sentence.strip().lower())
            for word in word_list:
                if word in word_dict:
                    word_dict[word]["freq"] += 1
                else:
                    word_dict[word] = {"id": newWordID, "freq": 1}
                    newWordID += 1
    return newWordID

wordID = count_words(train_data, wordID)
count_words(val_data, wordID)

with open("word_dict.pkl", 'wb') as f:
    pickle.dump(word_dict, f)
