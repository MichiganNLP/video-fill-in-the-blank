import torch
import torch.nn as nn
from transformers import BertModel

class multi_modal_model(nn.Module):
    def __init__(self, bert_model, V_D_in, embedding_size):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(multi_modal_model, self).__init__()
        self.video_embedding = nn.Linear(V_D_in, embedding_size)
        self.text_embedding = BertModel.from_pretrained('bert-base-uncased').get_input_embeddings()
        self.bert_model = bert_model

    def forward(self, text_feature, video_feature, attention_mask, segment_mask):
        video_feature_embeddings = self.video_embedding(video_feature)
        text_feature_embeddings = self.text_embedding(text_feature)

        input_feature = torch.cat([text_feature_embeddings, video_feature_embeddings], dim=1)
        out = self.bert_model(inputs_embeds=input_feature, attention_mask=attention_mask, token_type_ids=segment_mask)
        return out