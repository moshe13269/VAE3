import torch.nn as nn
import torch


def normal_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)


class FineTuning(nn.Module):
    def __init__(self):
        super(FineTuning, self).__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512*16, 13)

    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])

    def forward(self, inputs):
        return self.fc(inputs.view(inputs.shape[0], 512*16))


if __name__ == '__main__':
    a = FineTuning()
    m = FineTuning().float()
    d = torch.rand(10, 512, 4, 4).float()
    x = m(d)
    print(x.shape)
