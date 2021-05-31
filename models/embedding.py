import torch.nn as nn

from utils.conv_block import ConvBlock


class Embedding(nn.Module):
    """
    Model as described in the reference paper,
    """

    def __init__(self, in_channel=3, z_dim=64):
        super().__init__()
        self.block1 = ConvBlock(in_channel, z_dim, 3, max_pool=2)
        self.block2 = ConvBlock(z_dim, z_dim, 3, max_pool=2)
        self.block3 = ConvBlock(z_dim, z_dim, 3, max_pool=None, padding=1)
        self.block4 = ConvBlock(z_dim, z_dim, 3, max_pool=None, padding=1)

        self.init_params()

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        return out

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
