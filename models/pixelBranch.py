import torch
import torch.nn as nn

class RboexsPredictor(nn.Module):
    def __init__(self):
        super(RboexsPredictor, self).__init__()
        """
        Rbox,6个通道，是否是目标，目标的top，bottom，left， right，以及方向
        输入shape（[1, 256, 56, 56]）
        """
        self.conv1 = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(256, 4, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        F_score = self.sigmoid(self.conv1(x))
        geo = self.sigmoid(self.conv2(x))
        angle = self.sigmoid(self.conv3(x))
        F_geometry = torch.cat([geo, angle], 1)
        return F_score, F_geometry


class AttentionHeatMap(nn.Module):
    def __init__(self):
        super(AttentionHeatMap, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.conv1(x))