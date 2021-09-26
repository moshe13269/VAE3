import torch
import torch.nn.functional as F


def lsd(spec, pred_spec):
    spec = spec + 10 ** -10
    pred_spec = pred_spec + 10 ** -10
    return torch.sqrt(torch.mean(torch.square(torch.log10(spec, pred_spec) * 10)))


def kl_div(spec, pred_spec):
    return F.kl_div(F.log_softmax(spec, 0), F.softmax(pred_spec, 0), reduction="none").mean()


