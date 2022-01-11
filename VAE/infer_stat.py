import torch
import torch.nn as nn
from VAE.Encoder import Encoder as Encoder
from utils.dataloader import Dataset
import time
import os.path
from torch.nn import functional as F


file = open("/home/moshelaufer/PycharmProjects/VAE2/data/VAE/fine_tuning/process_state_encoder_2.txt", "a")
device = torch.device('cuda:1')

path2encoder = "/home/moshelaufer/PycharmProjects/VAE2/data/VAE/fine_tuning/model_fine_tune.pt"

encoder = Encoder()

if os.path.isfile(path2encoder):
    checkpoint = torch.load(path2encoder)
    encoder.load_state_dict(checkpoint['model_state_dict'])

encoder.to(device)
encoder.eval()

mse_criterion = nn.MSELoss().to(device)
ce_criterion = nn.CrossEntropyLoss().to(device)

dataset = Dataset(
    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0/",
    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0.csv",
    model_type='vae_fine_tuning', train=0)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1,
                                          pin_memory=True, drop_last=True)
print(len(data_loader.dataset))

loss_tot = 0.0
start_time = time.time()

a0_counter = 0
a1_counter = 0
a2_counter = 0
a3_counter = 0.0
counter = 0
for batch_num, data in enumerate(data_loader):
    # if batch_num % 200 == 0:
    #     print("sum samples = {} ".format(batch_num))
    spec = data[0]
    label = data[1]
    spec = spec.to(device)
    label_pred = encoder(spec)

    if batch_num == 10000:
        break;
    if label_pred[:, :4].squeeze().detach().cpu().numpy().argmax()/4. == label[:, :1].squeeze()/4:
        a0_counter += 1
    if label_pred[:, 4:8].squeeze().detach().cpu().numpy().argmax()/4. == label[:, 1:2].squeeze()/4:
        a1_counter += 1
    if label_pred[:, 8:10].squeeze().detach().cpu().numpy().argmax()/1 == 0:
        a2 = 0.
    else:
        a2 = 1.
    if a2 == label[:, 2:3].squeeze().detach().cpu().numpy():
        a2_counter += 1
    # print(label_pred[:, 8:10].squeeze().detach().cpu().numpy().argmax()/1, label[:, 2:3].squeeze().detach().cpu().numpy())
    a3_counter += mse_criterion(F.relu(label_pred[:, 10:].squeeze().detach().cpu()), label[:, 3:].squeeze())
    counter += 1

print("[osc2waveform accuracy %f] [lfo1waveform accuracy %f] [lfo1destination accuracy %f] [vector L2 Error %f] "
      "fine_tune"
      % (a0_counter/counter, a1_counter/counter, a2_counter/counter, a3_counter/counter))