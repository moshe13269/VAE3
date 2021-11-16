import numpy as np
import torch
import torch.nn.functional as F


v1 = np.asarray([0.0, 0.25, 0.5, 0.75])
v2 = np.asarray([0.0, 0.43])


def denormalized_vector(vector):
    vector[0] = v1[np.argmin(np.abs(v1 - vector[0]))]
    vector[1] = v1[np.argmin(np.abs(v1 - vector[1]))]
    vector[2] = v2[np.argmin(np.abs(v2 - vector[2]))]
    return vector


def convert_label4input(label):
    new_label = torch.zeros(6)
    new_label[0] = torch.LongTensor([int(label[0] * 4)])
    new_label[1] = torch.LongTensor([int(label[1] * 4)])
    if label[2] == 0.43:
        new_label[2] = torch.LongTensor([1])
    else:
        new_label[2] = torch.LongTensor([0])
    new_label[3] = label[3]
    new_label[4] = label[4]
    new_label[5] = label[5]
    return new_label


def convert_label4output(label):
    label = torch.from_numpy(label)
    new_label = torch.zeros(6)
    # print(label[:4], label[4:8])
    new_label[0] = F.softmax(label[:4], dim=0).argmax().item()/4
    new_label[1] = F.softmax(label[4:8], dim=0).argmax().item()/4
    value = F.softmax(label[8:10], dim=0).argmax().item()
    if value == 0:
        new_label[2] = 0
    else:
        new_label[2] = 0.43
    new_label[3] = label[10]
    new_label[4] = label[11]
    new_label[5] = label[12]
    return new_label


def convert_label4gan(label):
    one_hot_label = [0]*13
    one_hot_label[int(label[0] * 4)] = 1
    one_hot_label[int(label[1] * 4) + 4] = 1
    if label[2] == 0 or label[2] == 0.0:
        one_hot_label[8] = 1
    else:
        one_hot_label[9] = 1
    one_hot_label[10] = label[3]
    one_hot_label[11] = label[4]
    one_hot_label[12] = label[5]
    noise = torch.Tensor(torch.normal(torch.zeros(499), 0.1))
    return torch.cat((noise, torch.Tensor(one_hot_label))).unsqueeze(1).unsqueeze(2)
    # return torch.Tensor(one_hot_label)



