import torch
from transformers import AutoTokenizer


def batchPadding(batch, model_name: str, tokenizer):
    batch_size = len(batch)

    textFeatures = []
    videoFeatures = []
    labels = []
    mask_positions = []
    keys = []

    max_text_len = 0
    max_video_len = 0
    video = None
    for i in range(batch_size):
        data = batch[i]
        text = torch.tensor(data[0])
        video = data[1]
        labels.append(data[2])
        mask_positions.append(data[3])
        keys.append(data[4])

        textFeatures.append(text)
        videoFeatures.append(video)

        total_text_len = len(text)
        total_video_len = video.shape[0]
        if total_text_len > max_text_len:
            max_text_len = total_text_len
        if total_video_len > max_video_len:
            max_video_len = total_video_len

    text_tensor = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    video_tensor = torch.zeros(batch_size, max_video_len, video.shape[1], dtype=torch.float)

    segments_tensor = torch.cat([torch.zeros(batch_size, max_text_len, dtype=torch.long),
                                 torch.ones(batch_size, max_video_len, dtype=torch.long)], dim=1)
    attention_mask = torch.zeros(batch_size, max_text_len + max_video_len)
    # `-100` is the default `ignore_index` value for CrossEntropyLoss.
    masked_lm_labels = torch.ones(batch_size, max_text_len + max_video_len, dtype=torch.long) * -100
    position_embedding = torch.cat(
        [torch.arange(max_text_len, dtype=torch.long), torch.arange(max_video_len, dtype=torch.long)], dim=0)
    position_embedding = position_embedding.view(1, -1).repeat(batch_size, 1)

    for i in range(batch_size):
        text = textFeatures[i]
        video = videoFeatures[i]
        text_len = len(text)
        video_len = video.shape[0]

        # The input to the transformer is gonna be:
        # [CLS] t_1 ... t_n pad ... pad [SEP] v_1 ... v_m pad ... pad [SEP]

        text_tensor[i, :text_len - 1] = text[:-1]
        text_tensor[i, -1] = text[-1]

        video_tensor[i, :video_len] = video

        attention_mask[i, :text_len - 1] = 1
        attention_mask[i, max_text_len - 1:max_text_len + video_len] = 1

        masked_lm_labels[i, mask_positions[i]] = tokenizer.convert_tokens_to_ids(labels[i])

    return text_tensor, video_tensor, attention_mask, segments_tensor, labels, mask_positions, masked_lm_labels, \
           position_embedding, keys
