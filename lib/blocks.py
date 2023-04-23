import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, pre_norm=True):
        super().__init__()
        self.pre_norm = nn.BatchNorm2d(in_channels) if pre_norm else None

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        residual = x
        x = self.pre_norm(x) if self.pre_norm else x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.conv2(x)
        return x + residual
    
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads, pre_norm=True):
        super().__init__()
        self.pre_norm = nn.BatchNorm2d(in_channels) if pre_norm else None
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self.attn = nn.MultiheadAttention(7056, num_heads)
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        N, C, H, W = x.shape
        residual = x
        x = self.pre_norm(x) if self.pre_norm else x
        x = self.conv1(x)
        x = x.flatten(2)
        x = self.attn(x, x, x, need_weights=False)[0]
        x = x.reshape(N, C, H, W)
        x = self.norm(x)
        x = self.conv2(x)
        return x + residual
    
class FoundationTransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads, pre_norm=True):
        super().__init__()
        self.res = ResidualBlock(in_channels, pre_norm)
        self.attn = AttentionBlock(in_channels, num_heads)

    def forward(self, x):
        x = self.res(x)
        x = self.attn(x)
        return x
    
if __name__ == '__main__':
    AttentionBlock(32)(torch.randn(1, 32, 84, 84))