from unittest import TestCase

from transformers import AutoTokenizer

from lqam.data_module import QGenDataset


class TestQGenDataset(TestCase):
    def test_dataset_format(self):
        train_dataset = QGenDataset("https://drive.google.com/uc?id=1hFnEFGLMurexpz9c3QOKAHZtMl0utzIJ"
                                    "&export=download",
                                    tokenizer=AutoTokenizer.from_pretrained("t5-base"))
        train_expected_first_item = {
            "masked_caption": "<extra_id_0> and one man are riding on the back of an elephant.",
            "label": "Two Kids",
        }
        train_actual_first_item = train_dataset[0]

        val_dataset = QGenDataset("https://drive.google.com/uc?id=1Fv5Yf79guD-95yNNGpFr-GHUMrNc-gSv"
                                  "&export=download",
                                  tokenizer=AutoTokenizer.from_pretrained("t5-base"))
        val_expected_first_item = {
            "masked_caption": "A man is laying on the floor with a ball underneath <extra_id_0> and beginning to roll "
                              "back and forth.",
            "label": "his neck",
        }
        val_actual_first_item = val_dataset[0]

        test_train_dataset = QGenDataset("https://drive.google.com/uc?id=1h-8ADZJDr32QgZMClQ6J1mvMWQY0Ahzx"
                                         "&export=download",
                                         tokenizer=AutoTokenizer.from_pretrained("t5-base"))
        test_expected_first_item = {
            "masked_caption": "A person carrying something on their hand is walking outside of the room while another "
                              "person is reaching for <extra_id_0>.",
            "label": "the hallway",
        }
        test_actual_first_item = test_train_dataset[0]
        
        self.assertEqual(train_expected_first_item, train_actual_first_item)
        self.assertEqual(val_expected_first_item, val_actual_first_item)
        self.assertEqual(test_expected_first_item, test_actual_first_item)
