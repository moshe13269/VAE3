import torch
from Decoder.Generator import Generator
from utils.dataloader import Dataset


device = torch.device('cuda:3')
batch_size = 100

dataset = Dataset(
    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0/",
    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0.csv")
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                          pin_memory=True, drop_last=True)
PATH = "/home/moshelaufer/PycharmProjects/VAE2/data/GAN/weight.pt"
generator = Generator()
generator.load_state_dict(torch.load(PATH))
generator.eval()
generator.to(device)
