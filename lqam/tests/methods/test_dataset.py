from unittest import TestCase

from transformers import AutoTokenizer

from lqam.methods.dataset import QGenDataset, URL_DATA_TEST, URL_DATA_TRAIN, URL_DATA_VAL


class TestQGenDataset(TestCase):
    def test_dataset_format(self):
        train_dataset = QGenDataset(URL_DATA_TRAIN, tokenizer=AutoTokenizer.from_pretrained("t5-base"))
        train_expected_first_item = {
            "masked_caption": "_____ wearing harnesses using ropes to climb up a rock slope.",
            "label": "People",
        }
        train_actual_first_item = train_dataset[0]

        val_dataset = QGenDataset(URL_DATA_VAL, tokenizer=AutoTokenizer.from_pretrained("t5-base"))
        val_expected_first_item = {
            "masked_caption": "A man is laying on the floor with a ball underneath _____ and beginning to roll "
                              "back and forth.",
            "label": "his neck",
        }
        val_actual_first_item = val_dataset[0]

        test_train_dataset = QGenDataset(URL_DATA_TEST, tokenizer=AutoTokenizer.from_pretrained("t5-base"))
        test_expected_first_item = {
            "masked_caption": "A person carrying something on their hand is walking outside of the room while another "
                              "person is reaching for _____.",
            "label": "the hallway",
        }
        test_actual_first_item = test_train_dataset[0]
        
        self.assertEqual(train_expected_first_item, train_actual_first_item)
        self.assertEqual(val_expected_first_item, val_actual_first_item)
        self.assertEqual(test_expected_first_item, test_actual_first_item)
