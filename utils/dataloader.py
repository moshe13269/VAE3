from torch.utils.data import Dataset
import torch
from os import listdir
import os
from os.path import isfile, join
import numpy as np
from scipy import signal
from scipy.io import wavfile
import pickle
import pandas as pd
from utils.util import convert_label4input
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import numpy
import librosa


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


class Dataset(Dataset):
    def __init__(self, path2data, path2csv, train=1):
        self.train = train
        self.path2data = path2data
        list_files = next(os.walk(path2data))[2]
        if self.train:
            self.path_list = [join(path2data, file) for file in list_files \
                              if int(file.replace('.wav', '')) % 7 != 0]
        else:
            self.path_list = [join(path2data, file) for file in list_files
                              if int(file.replace('.wav', '')) % 7 == 0]

        self.csv_df = pd.read_csv(path2csv, skipinitialspace=True)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        try:
            fs, x = wavfile.read(join(self.path2data, str(index)+'.wav')) # self.path_list[index])
        except UnboundLocalError:
            print(self.path_list[index])
        f, t, Zxx = signal.stft(x*100, fs, nperseg=128, nfft=511, window='hamming')
        # f, t, Zxx = signal.stft(x, fs, nperseg=64, nfft=1023, window='hamming')
        Zxx = np.log(np.abs(Zxx) + 1)
        Zxx = torch.tensor(Zxx[:, :256])
        Zxx = Zxx.unsqueeze(0)
        # Zxx = librosa.amplitude_to_db(Zxx,  amin=1e-05) #ref=np.max,
        # Zxx = ((Zxx - Zxx.mean()) / Zxx.std())
        label = list(self.csv_df.loc[index])[1:]
        if self.train:
            label = convert_label4input(label)
        return Zxx, label


# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     dataset = Dataset("/home/moshelaufer/Documents/TalNoise/TAL14.07.2021/sounds_constADSR_res0/",
#                           "/home/moshelaufer/Documents/TalNoise/TAL14.07.2021/20210713_data_150k_constADSR_res0.csv")
#     data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
#     c = 0
#     cur_max = 0
#     arr_max = []
#     # for i in range(len(data_loader)):
#     for batch_num, data in enumerate(data_loader):
#         print(data[1])
#         plt.imshow(data[0].squeeze())