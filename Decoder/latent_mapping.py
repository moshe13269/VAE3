import torch.nn as nn
import torch
from torch.nn import functional as F


def normal_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        # torch.nn.init.normal_(m,0.0, 1)
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
        # m.bias.data.zero_()


class Latent(nn.Module):
    def __init__(self):
        super(Latent, self).__init__()
        self.conv1 = nn.ConvTranspose2d(13, 256, 4)
        self.bnm1 = nn.BatchNorm2d(num_features=256, momentum=0.1)
        self.conv2 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.conv3 = nn.ConvTranspose2d(256, 512, 3, padding=1)
        self.conv4 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.bnm4 = nn.BatchNorm2d(num_features=512, momentum=0.1)

    def forward(self, x):
        x = F.relu(self.bnm1(self.conv1(x)))
        x = F.relu(self.bnm1(self.conv2(x)))
        x = F.relu(self.bnm4(self.conv3(x)))
        x = F.relu(self.bnm4(self.conv4(x)))
        return x


# d = Latent()
# # d.weight_init()
# s = torch.rand(3, 13, 1, 1).float()
# print(d(s).shape)

