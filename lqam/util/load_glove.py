import torch
import numpy as np
import pickle

vectors = {}

with open(f'data/GloVe/glove.6B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        vect = np.array(line[1:]).astype(np.float)
        vect = torch.tensor(vect).unsqueeze(0)
        vectors[word] = vect

pickle.dump(vectors, open(f'data/GloVe/glove_embed.pkl', 'wb'))
