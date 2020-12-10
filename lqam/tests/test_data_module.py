from unittest import TestCase

from transformers import AutoTokenizer

from lqam.data_module import QGenDataset


class TestQGenDataset(TestCase):
    def test_dataset_format(self):
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        dataset = QGenDataset("https://drive.google.com/uc?id=1-JRsjFzP3Qmjti_w8ILV06msXjw4OXoB&export=download",
                              tokenizer=tokenizer)
        masked_caption = "A person is cleaning a window with <extra_id_0>."
        label = "a long window wiper"
        expected_first_item = {
            "masked_caption_ids": dataset._tokenize(masked_caption),
            "label_ids": dataset._tokenize(label),
        }
        actual_first_item = dataset[0]
        self.assertEqual(expected_first_item.keys(), actual_first_item.keys())
        for k in expected_first_item.keys():
            self.assertTrue((expected_first_item[k] == actual_first_item[k]).all())
