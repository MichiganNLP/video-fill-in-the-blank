from unittest import TestCase

import spacy
import torch

from lqam.methods.decoding import compute_noun_phrase_indices


class TestComputeNounPhraseMask(TestCase):
    def test_compute_noun_phrase_mask(self):
        spacy_model = spacy.load("en_core_web_lg")
        phrases = ["a dog and", "hello",
                   "sad", "happy",
                   "a cat and", "a dog"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = compute_noun_phrase_indices(spacy_model, phrases, batch_size=3, num_return_sequences=2, device=device)
        answer = [0, 2, 5]
        self.assertListEqual(mask.tolist(), answer)
