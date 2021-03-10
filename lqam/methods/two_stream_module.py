from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from overrides import overrides
from transformers import AutoModelForSeq2SeqLM, T5Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Stack

from lqam.methods.t5_visual_module import T5AndVisual, _combine_attention_masks


class TwoStreamEncoder(T5Stack):
    def __init__(self, t5stack: T5Stack, visual_size: int, pretrained_model_name: str = "t5-base") -> None:
        super().__init__(t5stack.config)
        self.text_stream = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name).encoder

        visual_config = T5Config.from_pretrained(pretrained_model_name, num_layers=1)
        self.visual_stream = AutoModelForSeq2SeqLM.from_config(visual_config).encoder

        self.embed_video = nn.Linear(visual_size, visual_config.d_model)

    @overrides
    def forward(self, text_input_ids: torch.Tensor, visual: torch.Tensor,  # noqa
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple[torch.Tensor, ...]]:
        text_feature = self.text_stream(text_input_ids, attention_mask=attention_mask).last_hidden_state

        visual_embedding = self.embed_video(visual)
        video_feature = self.visual_stream(inputs_embeds=visual_embedding,
                                           attention_mask=visual_attention_mask).last_hidden_state

        embedding = torch.cat([text_feature, video_feature], dim=1)

        attention_mask = _combine_attention_masks(attention_mask, visual_attention_mask)

        return super().forward(inputs_embeds=embedding, attention_mask=attention_mask, **kwargs)


class TwoStream(T5AndVisual):
    _keys_to_ignore_on_load_missing = (T5AndVisual._keys_to_ignore_on_load_missing  # noqa
                                       + [r"^encoder\.text_stream\.", r"^encoder\.visual_stream\."])

    def __init__(self, config: T5Config, visual_size: int, pretrained_model_name: str = "t5-base") -> None:
        super().__init__(config, visual_size)
        self.encoder = TwoStreamEncoder(self.encoder, visual_size, pretrained_model_name=pretrained_model_name)
