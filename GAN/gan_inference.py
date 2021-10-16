import torch
from Decoder.Discriminator import Discriminator
from Decoder.Generator import Generator
from utils.dataloader import Dataset
import matplotlib.pyplot as plt


device = torch.device('cuda:3')
batch_size = 1

dataset = Dataset(
    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0/",
    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0.csv")
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                          pin_memory=True, drop_last=True)

PATH = "/home/moshelaufer/PycharmProjects/VAE2/data/GAN/weight.pt"

generator = Generator()
generator.load_state_dict(torch.load(PATH)['G_state_dict'])
generator.to(device)

for i, data in enumerate(data_loader):
    fake = generator(data[1].to(device), torch.ones((1, 512, 4, 4)).to(device))
    spec = data[0].squeeze().cpu().numpy()
    fake = fake.squeeze().cpu().numpy()
    plt.figure()
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(spec)
    axarr[1].imshow(fake)

