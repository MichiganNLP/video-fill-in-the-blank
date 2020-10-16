import pickle

with open('train.pkl', 'rb') as f:
    train_data = pickle.load(f)

word_dict = {}
more_than_one_token = 0
for t in train_data:
    label = t[2]
    if len(label > 0):
        more_than_one_token += 1

print(more_than_one_token)