import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()
        inplanes = 512
        outplans = 256
        self.conv_1x1_1 = nn.Conv2d(512, outplans, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(outplans)

        self.conv_3x3_3 = nn.Conv2d(512, outplans, kernel_size=3, stride=1, padding=3, dilation=3)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(outplans)

        self.conv_3x3_6 = nn.Conv2d(512, outplans, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_6 = nn.BatchNorm2d(outplans)

        self.conv_3x3_9 = nn.Conv2d(512, outplans, kernel_size=3, stride=1, padding=9, dilation=9)
        self.bn_conv_3x3_9 = nn.BatchNorm2d(outplans)

        self.conv_3x3_12 = nn.Conv2d(512, outplans, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_12 = nn.BatchNorm2d(outplans)

        self.conv_3x3_15 = nn.Conv2d(512, outplans, kernel_size=3, stride=1, padding=15, dilation=15)
        self.bn_conv_3x3_15 = nn.BatchNorm2d(outplans)

        self.conv_3x3_18 = nn.Conv2d(512, outplans, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_18 = nn.BatchNorm2d(outplans)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_pool = nn.Conv2d(512, outplans, kernel_size=1)

        self.conv_1x1_3 = nn.Conv2d(2048, 256, kernel_size=1) #
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_6 = F.relu(self.bn_conv_3x3_6(self.conv_3x3_6(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_9 = F.relu(self.bn_conv_3x3_9(self.conv_3x3_9(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_12 = F.relu(self.bn_conv_3x3_12(self.conv_3x3_12(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_15 = F.relu(self.bn_conv_3x3_15(self.conv_3x3_15(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_18 = F.relu(self.bn_conv_3x3_18(self.conv_3x3_18(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        # avg_img = self.avg_pool(feature_map)
        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = self.conv_1x1_pool(out_img) # (shape: (batch_size, 256, 1, 1))
        # out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear", align_corners=True)
        out = torch.cat([out_1x1, out_3x3_3, out_3x3_6, out_3x3_9, out_3x3_12, out_3x3_15, out_3x3_18, out_img], 1) 
        # out is 1/16
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))

        return out

