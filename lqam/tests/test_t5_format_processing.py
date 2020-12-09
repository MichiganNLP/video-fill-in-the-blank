from unittest import TestCase

from transformers import AutoTokenizer

from lqam.t5_format_processing import compute_blank_map, fill_in_the_blanks, is_extra_token


class T5FormatProcessingTest(TestCase):
    def test_is_extra_token_id_0(self):
        self.assertTrue(is_extra_token("<extra_id_0>"))

    def test_is_extra_token_false(self):
        self.assertFalse(is_extra_token("hello"))

    def test_compute_blank_map_with_input_tokens(self):
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        input_tokens = tokenizer.tokenize("The <extra_id_0> walks in <extra_id_1> park.")
        generated_ids = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2> them <extra_id_3>")["input_ids"]

        expected_blank_map = {"<extra_id_0>": ["▁cute", "▁dog"], "<extra_id_1>": ["▁the"]}
        actual_blank_map = compute_blank_map(generated_ids, tokenizer, input_tokens)
        self.assertEqual(expected_blank_map, actual_blank_map)

    def test_compute_blank_map_without_input_tokens(self):
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        generated_ids = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2> them <extra_id_3>")["input_ids"]

        expected_blank_map = {"<extra_id_0>": ["▁cute", "▁dog"], "<extra_id_1>": ["▁the"], "<extra_id_2>": ["▁them"],
                              "<extra_id_3>": [tokenizer.eos_token]}
        actual_blank_map = compute_blank_map(generated_ids, tokenizer)
        self.assertEqual(expected_blank_map, actual_blank_map)

    def test_fill_in_the_blanks(self):
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        input_tokens = tokenizer.tokenize("The <extra_id_0> walks in <extra_id_1> park.")
        blank_map = {"<extra_id_0>": ["▁cute", "▁dog"], "<extra_id_1>": ["▁the"]}

        expected_filled_text = "The cute dog walks in the park."
        actual_filled_text = tokenizer.convert_tokens_to_string(list(fill_in_the_blanks(input_tokens, blank_map)))
        self.assertEqual(expected_filled_text, actual_filled_text)
