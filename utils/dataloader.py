from torch.utils.data import Dataset
import torch
import os
from os.path import join
import numpy as np
from scipy import signal
from scipy.io import wavfile
import pickle
import pandas as pd
from utils.util import convert_label4input, convert_label4gan


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


class Dataset(Dataset):
    def __init__(self, path2data, path2csv, model_type='vae', train=1):
        self.model_type = model_type
        self.train = train
        self.path2data = path2data
        self.path2csv = path2csv
        self.csv = self.open_csv()
        list_files = next(os.walk(path2data))[2]
        if self.train:
            self.path_list = [join(path2data, file) for file in list_files \
                              if int(file.replace('.wav', '')) % 7 != 0]
        elif self.model_type == 'vae_fine_tuning':
            import random
            random.shuffle(list_files)
            self.path_list = list_files[:10000]
        else:
            self.path_list = [join(path2data, file) for file in list_files
                              if int(file.replace('.wav', '')) % 7 == 0]

    def open_csv(self):
        file = open(self.path2csv)
        csv = pd.read_csv(file)
        csv = csv.to_numpy()
        return csv

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        try:
            fs, x = wavfile.read(join(self.path2data, str(index) + '.wav'))  # self.path_list[index])
        except UnboundLocalError:
            print(self.path_list[index])
        f, t, Zxx = signal.stft(x * 100, fs, nperseg=128, nfft=511, window='hamming')
        # f, t, Zxx = signal.stft(x, fs, nperseg=64, nfft=1023, window='hamming')
        Zxx = np.log(np.abs(Zxx) + 1)
        Zxx = torch.tensor(Zxx[:, :256])
        z_min = Zxx.min()
        Zxx = (((Zxx - z_min) / (Zxx.max() - z_min)) * 2) - 1
        Zxx = Zxx.unsqueeze(0)
        # Zxx = Zxx/Zxx.sum() # for probability
        # Zxx = librosa.amplitude_to_db(Zxx,  amin=1e-05) #ref=np.max,
        # Zxx = ((Zxx - Zxx.mean()) / Zxx.std())
        label = self.csv[index:index+1, 1:]  # list(self.csv_df.loc[index])[1:]
        label = label.squeeze()
        if self.model_type == 'vae':
            return Zxx, torch.Tensor(torch.normal(mean=torch.zeros(10, 256, 8, 8)))
        if self.train or self.model_type == 'vae_fine_tuning':
            label = convert_label4input(label)
            return Zxx, label

        label = convert_label4gan(label)
        return label


# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#
#     dataset = Dataset("/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0",
#                       "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0.csv")
#     data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
#     c = 0
#     cur_max = 0
#     arr_max = []
#     # for i in range(len(data_loader)):
#     for batch_num, data in enumerate(data_loader):
#         print(data[1])
#         plt.imshow(data[0].squeeze())
