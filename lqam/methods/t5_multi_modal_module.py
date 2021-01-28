from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from overrides import overrides
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack


class TextVisualEncoder(nn.Module):
    def __init__(self, t5_stack: T5Stack, text_embedding: nn.Embedding, visual_size: int) -> None:
        super().__init__()
        self.t5_stack = t5_stack
        self.text_embedding = text_embedding
        self.video_embedding = nn.Linear(visual_size, self.text_embedding.embedding_dim)

    def forward(self, text_token_ids: torch.Tensor,
                **kwargs) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple[torch.Tensor, ...]]:
        text_embedding = self.text_embedding(text_token_ids)
        visual_embedding = self.video_embedding(kwargs.pop('visual'))
        embedding = torch.cat([text_embedding, visual_embedding], dim=1)
        return self.t5_stack(inputs_embeds=embedding, **kwargs)


class T5AndVisual(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, visual_size: int) -> None:
        super().__init__(config)
        self.visual_size = visual_size

    def set_encoder(self) -> None:
        self.encoder = TextVisualEncoder(self.encoder, self.encoder.get_input_embeddings(), self.visual_size)

    @overrides
    def forward(self, masked_caption_ids: Optional[torch.Tensor] = None, visual: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, *args,
                **kwargs) -> Union[Seq2SeqLMOutput, Tuple[torch.Tensor, ...]]:
        if "encoder_outputs" not in kwargs:
            kwargs["encoder_outputs"] = self.encoder(masked_caption_ids, visual=visual)
        return super().forward(labels=labels, *args, **kwargs)  # noqa
