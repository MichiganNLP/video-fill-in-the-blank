from unittest import TestCase

from transformers import AutoTokenizer

from lqam.data_module import QGenDataset


class TestQGenDataset(TestCase):
    def test_dataset_format(self):
        dataset = QGenDataset("https://drive.google.com/uc?id=1-JRsjFzP3Qmjti_w8ILV06msXjw4OXoB&export=download",
                              tokenizer=AutoTokenizer.from_pretrained("t5-base"))
        expected_first_item = {
            "masked_caption": "A person is cleaning a window with <extra_id_0>.",
            "label": "a long window wiper",
        }
        actual_first_item = dataset[0]
        self.assertEqual(expected_first_item, actual_first_item)
