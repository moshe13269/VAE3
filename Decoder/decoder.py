import torch.nn as nn
import torch
from torch.nn import functional as F


def normal_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        # torch.nn.init.normal_(m,0.0, 1)
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
        # m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.ups1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.max_pool = nn.MaxPool2d(2)

    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])

    def forward(self, vector):
        spec = vector

        return spec
