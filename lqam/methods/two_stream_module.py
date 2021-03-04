from typing import Any, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from overrides import overrides
from transformers import T5Config, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput 
from transformers.models.t5.modeling_t5 import T5Stack


def _combine_attention_masks(text_attention_mask: Optional[torch.Tensor] = None,
                             visual_attention_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    if text_attention_mask is not None and visual_attention_mask is not None:
        return torch.cat([text_attention_mask, visual_attention_mask], dim=1)
    else:
        assert text_attention_mask is None and visual_attention_mask is None, \
            "Can't set the text or visual attention mask as one is empty and the other one isn't."
        return None


class TwoStreamEncoder(T5Stack):
    def __init__(self, t5stack: T5Stack, visual_size: int, hidden_size: int) -> None:
        super().__init__(t5stack.config)
        self.text_stream = AutoModelForSeq2SeqLM.from_pretrained('t5-base').encoder
        self.visual_stream = AutoModelForSeq2SeqLM.from_pretrained('t5-base').encoder

        self.embed_text = nn.Linear(hidden_size, self.config.d_model)
        self.embed_video1 = nn.Linear(visual_size, self.visual_stream.config.d_model)
        self.embed_video2 = nn.Linear(hidden_size, self.config.d_model)

    @overrides
    def forward(self, text_token_ids: torch.Tensor, visual: torch.Tensor,  # noqa
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple[torch.Tensor, ...]]:
        text_feature = self.text_stream(text_token_ids,attention_mask=attention_mask).last_hidden_state
        
        visual_embedding = self.embed_video1(visual)
        video_feature = self.visual_stream(inputs_embeds=visual_embedding, attention_mask=visual_attention_mask).last_hidden_state
        
        text_embedding = self.embed_text(text_feature)
        video_embedding = self.embed_video2(video_feature)
        embedding = torch.cat([text_embedding, visual_embedding], dim=1)

        attention_mask = _combine_attention_masks(attention_mask, visual_attention_mask)

        return super().forward(inputs_embeds=embedding, attention_mask=attention_mask, **kwargs)


class TwoStream(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, visual_size: int) -> None:
        super().__init__(config)
        self.encoder = TwoStreamEncoder(self.encoder, visual_size, config.d_model)

    @overrides
    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, visual: Optional[torch.Tensor] = None,
                                      visual_attention_mask: Optional[torch.Tensor] = None,
                                      **kwargs) -> Mapping[str, Any]:
        output = super().prepare_inputs_for_generation(input_ids, **kwargs)  # noqa
        output["visual"] = visual
        output["visual_attention_mask"] = visual_attention_mask
        return output

    @overrides
    def forward(self, masked_caption_ids: Optional[torch.Tensor] = None, visual: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, **kwargs) -> Union[Seq2SeqLMOutput, Tuple[torch.Tensor, ...]]:
        if "encoder_outputs" not in kwargs:
            kwargs["encoder_outputs"] = self.encoder(text_token_ids=masked_caption_ids, visual=visual, attention_mask=attention_mask,
                                                     visual_attention_mask=visual_attention_mask, **kwargs)

        # The attention mask used in the encoder (the combined mask) is actually necessary for the decoder even when
        # providing "encoder_outputs". We could compute it once in the encoder, return it as one of its fields and use
        # it here. However, in `T5FillerModel.generative_step` we input the encoder outputs but without the mask
        # since its constructed from the `generate` output which in turn only returns certain fields (not the mask).
        attention_mask = _combine_attention_masks(attention_mask, visual_attention_mask)

        return super().forward(attention_mask=attention_mask, labels=labels, **kwargs)  # noqa