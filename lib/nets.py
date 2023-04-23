import torch
from torch import nn
import copy

try:
    from lib.blocks import ResidualBlock, AttentionBlock, FoundationTransformerBlock
except ImportError:
    from blocks import ResidualBlock, AttentionBlock, FoundationTransformerBlock

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        C, H, W = input_dim

        if H != 84:
            raise ValueError(f"Expecting input height: 84, got: {H}")
        if H != 84:
            raise ValueError(f"Expecting input width: 84, got: {W}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        
class BattleshipNet(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        C, _, _ = input_dim

        self.online = nn.Sequential(
            nn.Conv2d(C, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            FoundationTransformerBlock(32, 4, pre_norm=False),
            FoundationTransformerBlock(32, 4),
            nn.Conv2d(32, C, 1, 1, 1, bias=False),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

if __name__ == '__main__':
    BattleshipNet((4, 84, 84), 6)(torch.rand((1, 4, 84, 84)), "online")