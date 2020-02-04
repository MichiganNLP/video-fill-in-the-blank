import torch
import torch.nn as nn

class baseline_BOW_VF(nn.module):
    def __init__(self, D_in, H, V_D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(baseline_BOW_VF, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(H, V_D_in)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(2 * V_D_in, D_out)

    def forward(self, text_feature, video_feature):
        h_relu = self.relu1(self.linear1(x))
        word_feature_out = self.relu2(self.linear2(h_relu))
        fused_feature = torch.cat([word_feature_out, video_feature], dim=0)
        y_pred = self.linear3(fused_feature)
        return y_pred