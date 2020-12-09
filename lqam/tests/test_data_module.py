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
        expected_first_item = (tokenizer(masked_caption, return_tensors="pt")["input_ids"],
                               tokenizer(label, return_tensors="pt")["input_ids"])
        actual_first_item = dataset[0]
        self.assertTrue((expected_first_item[0] == actual_first_item[0]).all())
        self.assertTrue((expected_first_item[1] == actual_first_item[1]).all())
