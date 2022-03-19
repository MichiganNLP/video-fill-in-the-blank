from collections import Iterator, Mapping
from unittest import TestCase

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from lqam.methods.t5_format_processing import compute_blank_map, compute_first_blank, is_extra_token


class T5FormatProcessingTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_name = "t5-small"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)

    def test_is_extra_token_id_0(self):
        token_id = self.tokenizer("<extra_id_0>", add_special_tokens=False)["input_ids"][0]
        self.assertTrue(is_extra_token(token_id, self.tokenizer))

    def test_is_extra_token_false(self):
        token_id = self.tokenizer("hello", add_special_tokens=False)["input_ids"][0]
        self.assertFalse(is_extra_token(token_id, self.tokenizer))

    def _compare_iterators_of_dicts_of_tensors(self, actual_blank_map: Iterator[Mapping[int, torch.Tensor]],
                                               expected_blank_map: Iterator[Mapping[int, torch.Tensor]]) -> None:
        # For some reason, comparing these int tensors directly inside the list and dictionaries doesn't work.
        # I think it's because internally it tries to convert the comparison result tensor into a bool, and it fails.
        for d1, d2 in zip(expected_blank_map, actual_blank_map):
            self.assertEqual(set(d1.keys()), set(d2.keys()))
            for k, v in d1.items():
                self.assertTrue((v == d2[k]).all())  # noqa

    def test_compute_blank_map_with_input_tokens(self):
        masked_caption_ids = self.tokenizer(["The <extra_id_0> walks in <extra_id_1> park."],
                                            return_tensors="pt")["input_ids"]
        generated_ids = self.tokenizer(
            ["<extra_id_0> cute dog <extra_id_1> the <extra_id_2> them <extra_id_3>"], return_tensors="pt")["input_ids"]

        expected_blank_map = [{"<extra_id_0>": ["▁cute", "▁dog"], "<extra_id_1>": ["▁the"]}]
        expected_blank_map = [
            {self.tokenizer.convert_tokens_to_ids(k): torch.tensor(self.tokenizer.convert_tokens_to_ids(v))
             for k, v in blank_map_instance.items()}
            for blank_map_instance in expected_blank_map
        ]
        actual_blank_map = compute_blank_map(generated_ids, self.tokenizer, masked_caption_ids)
        self._compare_iterators_of_dicts_of_tensors(actual_blank_map, expected_blank_map)  # noqa

    def test_compute_blank_map_without_input_tokens(self):
        generated_ids = self.tokenizer(
            ["<extra_id_0> cute dog <extra_id_1> the <extra_id_2> them <extra_id_3>"], return_tensors="pt")["input_ids"]

        expected_blank_map = [{
            "<extra_id_0>": ["▁cute", "▁dog"],
            "<extra_id_1>": ["▁the"],
            "<extra_id_2>": ["▁them"],
        }]
        expected_blank_map = [
            {self.tokenizer.convert_tokens_to_ids(k): torch.tensor(self.tokenizer.convert_tokens_to_ids(v))
             for k, v in blank_map_instance.items()}
            for blank_map_instance in expected_blank_map
        ]
        actual_blank_map = compute_blank_map(generated_ids, self.tokenizer)
        self._compare_iterators_of_dicts_of_tensors(actual_blank_map, expected_blank_map)

    def test_compute_first_blank(self):
        generated_ids = self.tokenizer(
            ["<extra_id_0> cute dog <extra_id_1> the <extra_id_2> them <extra_id_3>"], return_tensors="pt")["input_ids"]

        expected_first_blank_ids = self.tokenizer(["cute dog"], add_special_tokens=False,
                                                  return_tensors="pt")["input_ids"][0]

        decoder_start_token_id = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).config.decoder_start_token_id
        extra_id_0 = self.tokenizer.convert_tokens_to_ids(["<extra_id_0>"])[0]
        extra_id_1 = self.tokenizer.convert_tokens_to_ids(["<extra_id_1>"])[0]
        actual_first_blank_ids = next(iter(compute_first_blank(generated_ids,
                                                               decoder_start_token_id=decoder_start_token_id,
                                                               extra_id_0=extra_id_0, extra_id_1=extra_id_1)))

        self.assertTrue((expected_first_blank_ids == actual_first_blank_ids).all())  # noqa
