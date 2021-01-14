from unittest import TestCase

from lqam.core.noun_phrases import create_spacy_model
from lqam.methods.decoding import arg_noun_phrase


class TestDecoding(TestCase):
    def test_arg_noun_phrase(self):
        phrases = ["a dog", "a dog barks",
                   "a dog barks", "a dog barks",
                   "a dog barks", "a dog",
                   "a dog", "a dog"]
        expected_noun_phrase_indices = [0, 0, 1, 0]
        spacy_model = create_spacy_model(prefer_gpu=True)
        actual_noun_phrase_indices = arg_noun_phrase(spacy_model, phrases, num_return_sequences=2)
        self.assertListEqual(list(actual_noun_phrase_indices), expected_noun_phrase_indices)
