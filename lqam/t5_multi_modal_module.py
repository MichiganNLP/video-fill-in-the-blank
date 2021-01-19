from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack
import pytorch_lightning as pl
import torch
import torch.nn as nn
from overrides import overrides
from lqam.t5_module import T5FillerModel

class NewEncoder(nn.Module):
    def __init__(self, t5_stack: T5Stack, text_embedding, visual_size: int) -> None:
        super().__init__()
        self.t5_stack = t5_stack
        self.text_embedding = text_embedding
        self.video_embedding = nn.Linear(visual_size, self.text_embedding.embedding_dim)

    def forward(self, text_token_ids, *args, **kwargs):
        text_embedding = self.text_embedding(text_token_ids)
        visual_embedding = self.video_embedding(kwargs['visual'])
        embedding = torch.cat([text_embedding, visual_embedding], dim=1)
        return self.t5_stack.forward(inputs_embeds=embedding)


class T5AndI3D(T5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        self.visual_size = kwargs['visual_size']
        del kwargs['visual_size']
        super().__init__(*args, **kwargs)
        self.text_embedding = self.encoder.get_input_embeddings()
        

    def set_encoder(self):
        self.encoder = NewEncoder(self.encoder, self.text_embedding, self.visual_size)
    
    @overrides
    def forward(self, masked_caption_ids, visual = None, labels = None, *args, **kwargs):
        if "encoder_outputs" in kwargs:
            encoder_output = kwargs["encoder_outputs"]
        else:
            encoder_output = self.encoder(masked_caption_ids, visual=visual)
        return super().forward(encoder_outputs=encoder_output, labels = labels, return_dict = True)