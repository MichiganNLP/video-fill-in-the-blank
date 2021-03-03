from unittest import TestCase

from transformers import AutoTokenizer

from lqam.methods.dataset import QGenDataset, URL_DATA_TEST, URL_DATA_TRAIN, URL_DATA_VAL


class TestQGenDataset(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.tokenizer = AutoTokenizer.from_pretrained("t5-base")

    def test_dataset_format_train(self):
        train_dataset = QGenDataset(URL_DATA_TRAIN, tokenizer=self.tokenizer)
        train_expected_first_item = {
            "masked_caption": "_____ wearing harnesses using ropes to climb up a rock slope.",
            "label": "People",
        }
        train_actual_first_item = train_dataset[0]
        self.assertEqual(train_expected_first_item, train_actual_first_item)

    def test_dataset_format_val(self):
        val_dataset = QGenDataset(URL_DATA_VAL, tokenizer=self.tokenizer)
        val_expected_first_item = {
            "masked_caption": "In a gym with someone spotting him, a man is lifting _____ performing squats in a squat "
                              "rack.",
            "label": "weights",
            "additional_answers": [
                ["a weight", "some weights", "a squat bar"],
                ["weight", "a bar"],
                ["weights", "a barbell"],
                ["weights", "a heavy load"],
                ["weights", "a barbell"],
                ["weights", "iron", "equipment"],
                ["weights", "a barbell", "a weight"],
                ["weights", "405 lb"],
            ]
        }
        val_actual_first_item = val_dataset[0]
        self.assertEqual(val_expected_first_item, val_actual_first_item)

    def test_dataset_format_test(self):
        test_train_dataset = QGenDataset(URL_DATA_TEST, tokenizer=self.tokenizer)
        test_expected_first_item = {
            "masked_caption": "A person carrying something on their hand is walking outside of the room while another "
                              "person is reaching for _____.",
            "label": "the hallway",
        }
        test_actual_first_item = test_train_dataset[0]
        self.assertEqual(test_expected_first_item, test_actual_first_item)
