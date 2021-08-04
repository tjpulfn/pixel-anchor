import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
    def __init__(self):
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = 3
    
    def forward(self, model_out, label):
        score_map, geo_map, cls_map = model_out
        num_classes = cls_map.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(cls_map, dim=2)[:, :, 0]