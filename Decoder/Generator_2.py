import torch.nn as nn
import torch
from torch.nn import functional as F
# from Decoder.latent_mapping import Latent as Latent


def normal_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        # torch.nn.init.normal_(m,0.0, 1)
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
        # m.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ups1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.max_pool = nn.MaxPool2d(2)
        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.latent_conv1 = nn.ConvTranspose2d(13, 256, 4)
        self.latent_bnm1 = nn.BatchNorm2d(num_features=256, momentum=0.1)
        self.latent_conv2 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.latent_conv3 = nn.ConvTranspose2d(256, 512, 3, padding=1)
        self.latent_conv4 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.latent_bnm4 = nn.BatchNorm2d(num_features=512, momentum=0.1)

        self.conv1 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.bnm1 = nn.BatchNorm2d(num_features=512, momentum=0.1)
        self.conv2 = nn.ConvTranspose2d(512, 256, 3, padding=1)
        self.bnm2 = nn.BatchNorm2d(num_features=256, momentum=0.1)
        self.conv0_to_spec = nn.ConvTranspose2d(256, 1, 1, padding=0)

        self.conv3 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.conv4 = nn.ConvTranspose2d(256, 128, 3, padding=1)
        self.bnm4 = nn.BatchNorm2d(num_features=128, momentum=0.1)
        self.conv1_to_spec = nn.ConvTranspose2d(128, 1, 1, padding=0)

        self.conv5 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.conv6 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.bnm6 = nn.BatchNorm2d(num_features=64, momentum=0.1)
        self.conv2_to_spec = nn.ConvTranspose2d(64, 1, 1, padding=0)

        self.conv7 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.conv8 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.bnm8 = nn.BatchNorm2d(num_features=32, momentum=0.1)
        self.conv3_to_spec = nn.ConvTranspose2d(32, 1, 1, padding=0)

        self.conv9 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.conv10 = nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.bnm10 = nn.BatchNorm2d(num_features=16, momentum=0.1)
        self.conv4_to_spec = nn.Conv2d(16, 1, 1, padding=0)

        self.conv11 = nn.ConvTranspose2d(16, 16, 3, padding=1)
        self.conv12 = nn.ConvTranspose2d(16, 16, 3, padding=1)
        self.bnm12 = nn.BatchNorm2d(num_features=16, momentum=0.1)
        self.conv5_to_spec = nn.ConvTranspose2d(16, 1, 1, padding=0)

        self.conv13 = nn.ConvTranspose2d(16, 1, 3, padding=1)
        self.fc = nn.Linear(256, 256)

    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])

    def to_spec(self, x):
        if x.shape[2] == 4:
            return F.relu(self.conv0_to_spec(x))
        elif x.shape[2] == 8:
            return F.relu(self.conv1_to_spec(x))
        elif x.shape[2] == 16:
            return F.relu(self.conv2_to_spec(x))
        elif x.shape[2] == 32:
            return F.relu(self.conv3_to_spec(x))
        elif x.shape[2] == 64:
            return F.relu(self.conv4_to_spec(x))
        elif x.shape[2] == 128:
            return F.relu(self.conv5_to_spec(x))
        elif x.shape[2] == 256:
            return F.relu(self.conv6_to_spec(x))

    def forward(self, x, epoch):
        x = F.relu(self.latent_bnm1(self.latent_conv1(x)))
        x = F.relu(self.latent_bnm1(self.latent_conv2(x)))
        x = F.relu(self.latent_bnm4(self.latent_conv3(x)))
        x = F.relu(self.latent_bnm4(self.latent_conv4(x)))

        x = F.relu(self.bnm1(self.conv1(x)))
        x = F.relu(self.bnm2(self.conv2(x)))
        # x = x * param[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        spec0 = x
        spec0 = self.to_spec(spec0)
        spec0 = self.ups1(spec0)
        x = self.ups1(x)

        x = F.relu(self.bnm2(self.conv3(x)))
        x = F.relu(self.bnm4(self.conv4(x)))
        spec1 = x
        spec1 = self.to_spec(spec1)
        if epoch < 10:
            spec1 = self.ups1(spec1+spec0)
        else:
            spec1 = self.ups1(spec1)
        x = self.ups1(x)

        x = F.relu(self.bnm4(self.conv5(x)))
        x = F.relu(self.bnm6(self.conv6(x)))

        spec2 = x
        spec2 = self.to_spec(spec2)
        if epoch < 20:
            spec2 = self.ups1(spec2 + spec1)
        else:
            spec2 = self.ups1(spec2)
        x = self.ups1(x)

        x = F.relu(self.bnm6(self.conv7(x)))
        x = F.relu(self.bnm8(self.conv8(x)))

        spec3 = x
        if epoch < 30:
            spec3 = self.to_spec(spec3 + spec2)
        else:
            spec3 = self.to_spec(spec3)
        spec3 = self.ups1(spec3)
        x = self.ups1(x)

        x = F.relu(self.bnm8(self.conv9(x)))
        x = F.relu(self.bnm10(self.conv10(x)))

        spec4 = x
        if epoch < 40:
            spec4 = self.to_spec(spec4 + spec3)
        else:
            spec4 = self.to_spec(spec4)
        spec4 = self.ups1(spec4)
        x = self.ups1(x)

        x = F.relu(self.bnm10(self.conv11(x)))
        x = F.relu(self.bnm12(self.conv12(x)))

        spec5 = x
        if epoch < 50:
            spec5 = self.to_spec(spec5 + spec4)
            spec5 = self.ups1(spec5)
        x = self.ups1(x)

        x = F.relu(self.conv13(x))

        if epoch < 50:
            x = torch.tanh(self.fc(x))
            return x + spec5
        x = torch.tanh(self.fc(x))
        return x


# d = Generator()
# # # d.weight_init()
# s = torch.rand(3, 13, 1, 1).float()
# print(d(s, 55).shape)

