import torch.nn as nn
import torch
# from torch.nn import functional as F


def normal_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.ups1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = nn.Conv2d(1, 16, 4, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1, stride=2)
        self.bnm16 = nn.BatchNorm2d(num_features=16, momentum=0.1)

        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.bnm32 = nn.BatchNorm2d(num_features=32, momentum=0.1)

        self.conv7 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv9 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.bnm64 = nn.BatchNorm2d(num_features=64, momentum=0.1)

        self.conv10 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv11 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.bnm128 = nn.BatchNorm2d(num_features=128, momentum=0.1)

        self.conv12 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv13 = nn.Conv2d(256, 256, 3, padding=1, stride=2)
        self.bnm256 = nn.BatchNorm2d(num_features=256, momentum=0.1)

        # decoder
        self.conv22 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.conv23 = nn.ConvTranspose2d(256, 256, 3, padding=1)

        self.conv24 = nn.ConvTranspose2d(256, 128, 3, padding=1)
        self.conv25 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.conv26 = nn.ConvTranspose2d(128, 128, 3, padding=1)

        self.conv27 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.conv28 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.conv29 = nn.ConvTranspose2d(64, 64, 3, padding=1)

        self.conv30 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.conv31 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.conv32 = nn.ConvTranspose2d(32, 32, 3, padding=1, stride=2)

        self.conv33 = nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.conv34 = nn.ConvTranspose2d(16, 16, 3, padding=1)
        self.conv35 = nn.ConvTranspose2d(16, 1, 3, padding=0)

        self.lrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])

    def forward(self, inputs, noise):
        x = self.bnm16(self.relu(self.conv1(inputs)))
        x = self.bnm16(self.relu(self.conv2(x)))
        x = self.bnm16(self.relu(self.conv3(x)))
        # layer1 = x

        x = self.bnm32(self.relu(self.conv4(x)))
        x = self.bnm32(self.relu(self.conv5(x)))
        x = self.bnm32(self.relu(self.conv6(x)))
        # layer2 = x

        x = self.bnm64(self.relu(self.conv7(x)))
        x = self.bnm64(self.relu(self.conv8(x)))
        x = self.bnm64(self.relu(self.conv9(x)))
        # layer3 = x

        x = self.bnm128(self.relu(self.conv10(x)))
        x = self.bnm128(self.relu(self.conv11(x)))
        # layer4 = x

        x = self.bnm256(self.relu(self.conv12(x)))
        x = self.bnm256(self.relu(self.conv13(x)))

        """Decoder"""
        x = torch.unsqueeze(x.sum(axis=-1), axis=-1) * x
        x = self.bnm256(self.lrelu(self.conv22(x + noise)))#+ layer5))
        x = self.bnm256(self.lrelu(self.conv23(x)))
        x = self.ups1(x)

        x = self.bnm128(self.lrelu(self.conv24(x)))# + layer4))
        x = self.bnm128(self.lrelu(self.conv25(x)))
        x = self.bnm128(self.lrelu(self.conv26(x)))  # + layer3))
        x = self.ups1(x)

        x = self.bnm64(self.lrelu(self.conv27(x)))
        x = self.bnm64(self.lrelu(self.conv28(x)))
        x = self.bnm64(self.lrelu(self.conv29(x))) # + layer2))
        x = self.ups1(x)

        x = self.bnm32(self.lrelu(self.conv30(x)))
        x = self.bnm32(self.lrelu(self.conv31(x)))
        x = self.bnm32(self.lrelu(self.conv32(x))) # + layer1))
        x = self.ups1(x)

        x = self.bnm16(self.lrelu(self.conv33(x)))
        x = self.bnm16(self.lrelu(self.conv34(x)))
        x = torch.tanh(self.conv35(x))
        return x

# a = VAE()
# m = VAE().float()
# d = torch.rand(10,1,256,256).float()
# f = torch.Tensor(torch.normal(mean=torch.zeros(10, 256, 8, 8)))
# x = m(d, f)
# print(x.shape)
