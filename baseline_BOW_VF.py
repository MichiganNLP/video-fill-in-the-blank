import torch
import torch.nn as nn

class baseline_BOW_VF(nn.Module):
    def __init__(self, W_D_in, V_D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(baseline_BOW_VF, self).__init__()
        self.linear1 = nn.Linear(W_D_in, V_D_in)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(2 * V_D_in, D_out)

    def forward(self, text_feature, video_feature):
        text_feature_out = self.relu1(self.linear1(text_feature))
        fused_feature = torch.cat([text_feature_out, video_feature], dim=1)
        y_pred = self.linear2(fused_feature)
        return y_pred