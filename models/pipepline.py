from pdb import set_trace
import torch
import torch.nn as nn
import torch.nn.functional as F
from aspp import ASPP
from resnet import ResNet, Bottleneck
from pixelBranch import RboexsPredictor, AttentionHeatMap
class Pixel_Anchor(nn.Module):
    def __init__(self) -> None:
        super(Pixel_Anchor, self).__init__()
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3])
        self.aspp = ASPP()
        self.rbox = RboexsPredictor()
        self.atten = AttentionHeatMap()
        self.conv8x = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv4x = nn.Sequential(
            nn.Conv2d(320, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
    
    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        min_block, mid_block, max_block = self.backbone(x)      # 通过backbone
        aspp_conv = self.aspp(max_block)    # 通过aspp模块，来自deeplab
        pixel_fpn = self._pixel_module(aspp_conv, min_block, mid_block, h, w)       # 通过fpn结构，将1/16大小的特征图进行多次上采样，最终与1/4大小特征图拼接
        F_score, F_geometry = self.rbox(pixel_fpn)
        atten_map = self.atten(pixel_fpn)
        return F_score, F_geometry, atten_map

    def _upsample(self, aspp_out, h, w, rate):
        return F.interpolate(aspp_out, size=(int(h / rate), int( w / rate)), mode="bilinear", align_corners=True)

    def _pixel_module(self, aspp_conv, min_block, mid_block, h=224, w=224):
        up1 = self._upsample(aspp_conv, h, w, rate=8)   # (batch_size, 256, h/8, w/8)
        fm8x = self.conv8x(torch.cat([up1, mid_block], 1))
        up2 = self._upsample(fm8x, h, w, rate=4)     # (batch_size, 256, h/4, w/4)
        fm4x = self.conv4x(torch.cat([up2, min_block], 1))
        return fm4x


aspp = Pixel_Anchor()
x = torch.randn(1, 3, 224, 224)
x = aspp(x)
for i in x:
    print(i.shape)