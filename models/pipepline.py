from pdb import set_trace
from typing import DefaultDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.aspp import ASPP
from models.resnet import ResNet, Bottleneck
from models.pixelBranch import RboexsPredictor, AttentionHeatMap

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


def base_convolutional(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )

class Pixel_Anchor(nn.Module):
    def __init__(self, num_cls) -> None:
        super(Pixel_Anchor, self).__init__()
        self.num_cls = num_cls
        self.box_outplanes = 3 * (self.num_cls + 9)
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3])
        self.aspp = ASPP()
        self.rbox = RboexsPredictor()
        self.atten = AttentionHeatMap()
        self.mish = Mish()
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
        self.s_box = nn.Sequential(
            nn.Conv2d(64, 3 * (self.num_cls + 9), kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3 * (self.num_cls + 9)),
        )
        self.s_box_conv = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.m_box = nn.Sequential(
            nn.Conv2d(384, 3 * (self.num_cls + 9), kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3 * (self.num_cls + 9)),
        )
        self.m_box_conv = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.l_box = nn.Sequential(
            nn.Conv2d(768, 3 * (self.num_cls + 9), kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3 * (self.num_cls + 9)),
        )
        self.l_box_conv = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        
    
    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        self.min_block, self.mid_block, self.max_block = self.backbone(x)      # 通过backbone, 1/4, 1/8, 1/16， s, m, l
        print("s size is :", self.min_block.shape)
        print("m size is :", self.mid_block.shape)
        print("l size is :", self.max_block.shape)
        aspp_conv = self.aspp(self.max_block)    # 通过aspp模块，来自deeplab
        F_score, geo_map, cls_map = self._pixel_module(aspp_conv, h, w)       # 通过fpn结构，将1/16大小的特征图进行多次上采样，最终与1/4大小特征图拼接
        # F_score, geo_map 计算loss， cls_map 1/4大小图
        
        conv = self._anchor_module(cls_map)     # 1/4 feature       # [1, 64, 56, 56]
        print(conv.shape, "321")
        conv_sbbox = self.s_box(conv)

        conv = self.s_box_conv(conv)
        conv = torch.cat([conv, self.mid_block], 1)
        print(conv.shape, "321")
        conv_mbbox = self.m_box(conv)

        conv = self.m_box_conv(conv)
        conv = torch.cat([conv, self.max_block], 1)
        print(conv.shape, "321")
        conv_lbbox = self.l_box(conv)

        return F_score, geo_map, (conv_sbbox, conv_mbbox, conv_lbbox)

    def _upsample(self, aspp_out, h, w, rate):
        return F.interpolate(aspp_out, size=(int(h / rate), int( w / rate)), mode="bilinear", align_corners=True)

    def _pixel_module(self, aspp_conv, h=224, w=224):
        up1 = self._upsample(aspp_conv, h, w, rate=8)   # (batch_size, 256, h/8, w/8)
        fm8x = self.conv8x(torch.cat([up1, self.mid_block], 1))
        up2 = self._upsample(fm8x, h, w, rate=4)     # (batch_size, 256, h/4, w/4)
        fm4x = self.conv4x(torch.cat([up2, self.min_block], 1))
        F_score, F_geometry = self.rbox(fm4x)
        atten_map = self.atten(fm4x)
        return F_score, F_geometry, atten_map

    def _anchor_module(self, atten_map):
        atten_exp = torch.exp(atten_map)
        fm4x_anchor = torch.mul(self.min_block, atten_exp)
        return fm4x_anchor
    
    def _local_module(self, aspp_conv, h=224, w=224):
        up1 = self._upsample(aspp_conv, h, w, rate=8)   # (batch_size, 256, h/8, w/8)
        fm8x = self.conv8x(torch.cat([up1, self.mid_block], 1))
        up2 = self._upsample(fm8x, h, w, rate=4)     # (batch_size, 256, h/4, w/4)
        fm4x = self.conv4x(torch.cat([up2, self.min_block], 1))
        F_score, geo_map, cls_map = self.rbox(fm4x)
        # atten_map = self.atten(fm4x)
        return F_score, geo_map, cls_map




aspp = Pixel_Anchor(2)
x = torch.randn(1, 3, 224, 224)
x = aspp(x)
for i in x:
    print(i.shape)