from unittest import TestCase

from lqam.core.noun_phrases import create_spacy_model
from lqam.methods.decoding import arg_noun_phrase


class TestDecoding(TestCase):
    def test_arg_noun_phrase(self):
        masked_caption = ["It is _____.", "It was _____.", "It will be _____.", "It has been _____."]
        phrases = [["a dog", "a dog and it barks"],
                   ["a dog and it barks", "a dog and it barks"],
                   ["a dog and it barks", "a dog"],
                   ["a dog", "a dog"]]
        expected_noun_phrase_indices = [0, 0, 1, 0]

        spacy_model = create_spacy_model(prefer_gpu=True)
        actual_noun_phrase_indices = list(arg_noun_phrase(spacy_model, masked_caption, phrases))

        self.assertListEqual(actual_noun_phrase_indices, expected_noun_phrase_indices)
