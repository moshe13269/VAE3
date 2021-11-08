import torch
from Decoder.Generator_2 import Generator as Generator
from utils.dataloader import Dataset
import matplotlib.pyplot as plt


device = torch.device('cuda:2')
batch_size = 1

dataset = Dataset(
    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0/",
    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0.csv")
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

PATH = "/home/moshelaufer/PycharmProjects/VAE2/data/GAN_2/weight.pt"

generator = Generator()
generator.load_state_dict(torch.load(PATH)['G_state_dict'])
generator.to(device)
generator.eval()

for i, data in enumerate(data_loader):
    vec = data[1].to(device)
    fake = generator(vec)
    spec = data[0].squeeze().cpu().detach().numpy()
    fake = fake.squeeze().cpu().detach().numpy()
    a=0
    # plt.figure()
    # f, axarr = plt.subplots(2, 1)
    # axarr[0].imshow(spec)
    # axarr[1].imshow(fake)
    a = 0

