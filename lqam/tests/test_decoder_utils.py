from unittest import TestCase
from lqam.decoder_utils import compute_noun_phrase_mask

import torch
import spacy


class TestComputeNounPhraseMask(TestCase):
    def test_compute_noun_phrase_mask(self):
        nlp = spacy.load("en_core_web_sm")
        phrases = ['a dog', 'hello', 'sad', 'happy']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask = compute_noun_phrase_mask(nlp, phrases, batch_size=2, num_return_sequences=2, device=device)
        answer = [True, False, True, False]
        self.assertListEqual(mask.tolist(), answer)
