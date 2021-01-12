from unittest import TestCase

from lqam.core.noun_phrases import create_spacy_model_for_noun_phrase_check
from lqam.methods.decoding import arg_noun_phrase


class TestComputeNounPhraseMask(TestCase):
    def test_compute_noun_phrase_mask(self):
        spacy_model = create_spacy_model_for_noun_phrase_check()
        phrases = ["a dog", "a dog barks",
                   "a dog barks", "a dog barks",
                   "a dog barks", "a dog",
                   "a dog", "a dog"]
        expected_noun_phrase_indices = [0, 0, 1, 0]
        actual_noun_phrase_indices = arg_noun_phrase(spacy_model, phrases, num_return_sequences=2)
        self.assertListEqual(list(actual_noun_phrase_indices), expected_noun_phrase_indices)
