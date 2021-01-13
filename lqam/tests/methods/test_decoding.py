from unittest import TestCase

from lqam.core.noun_phrases import SPACY_MODEL
from lqam.methods.decoding import arg_noun_phrase


class TestDecoding(TestCase):
    def test_arg_noun_phrase(self):
        phrases = ["a dog", "a dog barks",
                   "a dog barks", "a dog barks",
                   "a dog barks", "a dog",
                   "a dog", "a dog"]
        expected_noun_phrase_indices = [0, 0, 1, 0]
        actual_noun_phrase_indices = arg_noun_phrase(SPACY_MODEL, phrases, num_return_sequences=2)
        self.assertListEqual(list(actual_noun_phrase_indices), expected_noun_phrase_indices)
