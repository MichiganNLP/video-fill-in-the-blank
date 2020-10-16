import pickle

with open('/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/latest_data/multimodal_model/verb_data/test.pkl', 'rb') as f:
    train_data = pickle.load(f)

word_dict = {}
more_than_one_token = 0
for t in train_data:
    label = t[2]
    if len(label) > 1:
        more_than_one_token += 1

print(more_than_one_token)