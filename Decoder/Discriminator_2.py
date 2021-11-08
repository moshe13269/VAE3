import torch.nn as nn
import torch
from torch.nn import functional as F


def normal_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        # torch.nn.init.normal_(m,0.0, 1)
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
        # m.bias.data.zero_()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.max_pool = nn.MaxPool2d(2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.conv1 = nn.Conv2d(1, 16, 4, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bnm2 = nn.BatchNorm2d(num_features=16, momentum=0.1)

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.bnm4 = nn.BatchNorm2d(num_features=32, momentum=0.1)

        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.bnm6 = nn.BatchNorm2d(num_features=64, momentum=0.1)

        self.conv7 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.bnm8 = nn.BatchNorm2d(num_features=128, momentum=0.1)

        self.conv9 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1)
        self.bnm10 = nn.BatchNorm2d(num_features=256, momentum=0.1)

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bnm12 = nn.BatchNorm2d(num_features=512, momentum=0.1)

        self.fc1 = nn.Linear(4608, 658)
        self.fc2 = nn.Linear(658, 13)
        self.fc3 = nn.Linear(13, 1)

    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])

    def forward(self, x, epoch):
        # x0 = self.down(x)
        x = F.leaky_relu(self.bnm2(self.conv1(x)))
        x0 = x#self.down(x)
        if epoch < 10:
            x = F.leaky_relu(self.bnm2(self.conv2(x))+x0)
        else:
            x = F.leaky_relu(self.bnm2(self.conv2(x)))
        x = self.max_pool(x)

        # x1 = self.down(x)
        x = F.leaky_relu(self.bnm4(self.conv3(x)))
        x1 = x#self.down(x)
        if epoch < 20:
            x = F.leaky_relu(self.bnm4(self.conv4(x)) + x1)
        else:
            x = F.leaky_relu(self.bnm4(self.conv4(x)))
        x = self.max_pool(x) #+ x1

        # x2 = self.down(x)
        x = F.leaky_relu(self.bnm6(self.conv5(x)))
        x2 = x#self.down(x)
        if epoch < 30:
            x = F.leaky_relu(self.bnm6(self.conv6(x)) + x2)
        else:
            x = F.leaky_relu(self.bnm6(self.conv6(x)))
        x = self.max_pool(x) #+ x2

        # x3 = self.down(x)
        x = F.leaky_relu(self.bnm8(self.conv7(x)))
        x3 = x# self.down(x)
        if epoch < 40:
            x = F.leaky_relu(self.bnm8(self.conv8(x)) + x3)
        else:
            x = F.leaky_relu(self.bnm8(self.conv8(x)))
        x = self.max_pool(x) #+ x3

        # x4 = self.down(x)
        x = F.leaky_relu(self.bnm10(self.conv9(x)))
        x4 = x# self.down(x)
        if epoch < 50:
            x = F.leaky_relu(self.bnm10(self.conv10(x)) + x4)
        else:
            x = F.leaky_relu(self.bnm10(self.conv10(x)))
        x = self.max_pool(x) #+ x4

        # x5 = self.down(x)
        x = F.leaky_relu(self.bnm12(self.conv11(x)))
        x5 = x#self.down(x)
        if epoch < 60:
            x = F.leaky_relu(self.bnm12(self.conv12(x)) + x5)
        else:
            x = F.leaky_relu(self.bnm12(self.conv12(x)))
        x = self.max_pool(x) #+ x5

        # option: add residual function
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# d = Discriminator()
# # d.weight_init()
# s = torch.rand(3,1,256,256).float()
# print(d(s, 500))
# print(s)
# rr = nn.Upsample(scale_factor=0.5, mode='bilinear')
# print(rr(s))
# # print(F.interpolate(s, scale_factor=0.5, mode='bilinear').shape)