from unittest import TestCase
from lqam.decoder_utils import compute_noun_phrase_indices

import torch
import spacy


class TestComputeNounPhraseMask(TestCase):
    def test_compute_noun_phrase_mask(self):
        nlp = spacy.load("en_core_web_sm")
        phrases = ['a dog and', 'hello',
                   'sad', 'happy',
                   'a cat and', 'a dog']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask = compute_noun_phrase_indices(nlp, phrases, batch_size=2, num_return_sequences=2, device=device)
        answer = [2, 5]
        self.assertListEqual(mask.tolist(), answer)
