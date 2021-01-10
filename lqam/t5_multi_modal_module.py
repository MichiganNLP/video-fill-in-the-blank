from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack
import pytorch_lightning as pl
import torch
import torch.nn as nn
from overrides import overrides
from lqam.t5_module import T5FillerModel

class NewEncoder(nn.Module):
    def __init__(self, t5_stack: T5Stack) -> None:
        super().__init__()
        self.t5_stack = t5_stack

    def forward(self, inputs_embeds, *args, **kwargs):
        return self.t5_stack.forward(inputs_embeds=inputs_embeds, *args, **kwargs)


class T5AndI3D(T5ForConditionalGeneration):
    def __init__(self, visual_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_embedding = self.encoder.get_input_embeddings()
        self.encoder = NewEncoder(self.encoder)
        self.video_embedding = nn.Linear(visual_size, self.text_embedding.embedding_dim)

    @overrides
    def forward(self, masked_caption_ids, label_ids, *args, **kwargs):
        text_token_ids, visual = inputs
        text_embedding = self.text_embedding(text_token_ids)
        visual_embedding = self.video_embedding(visual)
        embedding = torch.cat([text_embedding, visual_embedding], dim=1)
        
        return super().forward(inputs_embeds=embedding, labels = labels, return_dict = True)