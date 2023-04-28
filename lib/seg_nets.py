import torch
from torch import nn
import torch.nn.functional as f

from lib.blocks import MetaResBlock, MetaFFCBlock, EfficientAttention, DAPPM

eps = 1e-3
momentum = 0.01

class MetaFFCFormer(nn.Module):
    def __init__(self, input_channels, num_classes, base_channels=32, spp_channels=64, head_channels=64):
        super().__init__()
        C = input_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(C, base_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.SyncBatchNorm(base_channels, eps=eps, momentum=momentum)
        self.res1 = MetaResBlock(base_channels, norm_input=False)
        self.ffcres1 = MetaFFCBlock(base_channels)

        self.pre_bn2 = nn.SyncBatchNorm(base_channels, eps=eps, momentum=momentum)
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.post_bn2 = nn.SyncBatchNorm(base_channels*2, eps=eps, momentum=momentum)
        self.res2 = MetaResBlock(base_channels*2, norm_input=False)
        self.ffcres2 = MetaFFCBlock(base_channels*2)

        self.pre_bn3_ = nn.SyncBatchNorm(base_channels*2, eps=eps, momentum=momentum)
        self.conv3_ = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.post_bn3_ = nn.SyncBatchNorm(base_channels*4, eps=eps, momentum=momentum)
        self.res3_ = MetaResBlock(base_channels*4, norm_input=False)

        self.ea3 = EfficientAttention(base_channels*2, base_channels*4, 4, 64, 64)
        self.res3 = MetaResBlock(base_channels*2)

        self.pre_bn4_ = nn.SyncBatchNorm(base_channels*4, eps=eps, momentum=momentum)
        self.conv4_ = nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.post_bn4_ = nn.SyncBatchNorm(base_channels*8, eps=eps, momentum=momentum)
        self.res4_ = MetaResBlock(base_channels*8, norm_input=False)
        self.ea4_ = EfficientAttention(base_channels*8, base_channels*2, 4, 64, 64)
        self.res4_2_ = MetaResBlock(base_channels*8)
        self.bn4_ = nn.SyncBatchNorm(base_channels*8, eps=eps, momentum=momentum)
        self.dappm_ = DAPPM(base_channels*8, spp_channels, base_channels*4)
        self.up_ = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.head_bn1 = nn.SyncBatchNorm(base_channels*6, eps=eps, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.head_conv1 = nn.Conv2d(base_channels*6, head_channels*2, kernel_size=3, padding=1, bias=False)
        self.head_bn2 = nn.SyncBatchNorm(head_channels*2, eps=eps, momentum=momentum)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.head_conv2 = nn.Conv2d(head_channels*2, head_channels, kernel_size=3, padding=1, bias=False)
        self.head_conv3 = nn.Conv2d(head_channels, num_classes, kernel_size=3, padding=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in reversed(list(self.modules())[1:]):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.zero_()
            if callable(getattr(module, '_init_weights', None)):
                module._init_weights()


    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.res1(x)
        x = self.ffcres1(x)
        
        # Layer 2
        x = self.pre_bn2(x)
        x = self.conv2(x)
        x = self.post_bn2(x)
        x = self.res2(x)
        x = self.ffcres2(x)

        # Layer 3
        x_ = self.pre_bn3_(x)
        x_ = self.conv3_(x_)
        x_ = self.post_bn3_(x_)
        x_ = self.res3_(x_)

        x = self.ea3(x, x_)
        x = self.res3(x)

        # Layer 4
        x_ = self.pre_bn4_(x_)
        x_ = self.conv4_(x_)
        x_ = self.post_bn4_(x_)
        x_ = self.res4_(x_)
        x_ = self.ea4_(x_, x)
        x_ = self.res4_2_(x_)
        x_ = self.bn4_(x_)
        x_ = self.dappm_(x_)
        x_ = self.up_(x_)

        x = torch.cat([x, x_], dim=1)

        # Head
        x = self.head_bn1(x)
        x = self.relu(x)
        x = self.head_conv1(x)
        x = self.head_bn2(x)

        x = self.up(x)
        x = self.relu(x)
        x = self.head_conv2(x)
        
        x = self.up(x)
        x = self.relu(x)
        x = self.head_conv3(x)

        return x