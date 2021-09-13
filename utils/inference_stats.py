import pandas as pd
import torch
from models.model_VAE_osc2 import VAE_osc2
# from models.model_Encoder import Encoder
from utils.dataloader import Dataset
import numpy as np
import os
from utils.util import denormalized_vector, convert_label4output
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Results:

    def __init__(self, path2save, path2encoder, path2vae, path2dataset, path2csv, vae=1):
        self.path2save = path2save
        self.device = torch.device('cuda:3')
        self.path2Encoder = path2encoder
        self.path2VAE = path2vae
        # self.Encoder = Encoder().float().to(self.device)
        self.VAE = VAE_osc2().float().to(self.device)
        self.path2csv = path2csv
        self.path2dataset = path2dataset
        self.vae = vae

    def load_weight_model(self):
        # self.Encoder.load_state_dict(torch.load(self.path2Encoder)['model_state_dict'])
        # self.Encoder.eval()
        self.VAE.load_state_dict(torch.load(self.path2VAE)['model_state_dict'])
        self.VAE.eval()

    def save_results2csv(self, np_array):
        columns = ['24.osc2waveform']#, '26.lfo1waveform', '32.lfo1destination',
                #   '30.lfo1amount', '28.lfo1rate', '3.cufoff'] #, '4.resonance'
        df = pd.DataFrame(np_array, columns=columns)
        if self.vae:
            df.to_csv(os.path.join(self.path2save,'vae.csv'))
        else:
            df.to_csv(os.path.join(self.path2save, 'encoder.csv'))
        print('csv file had been saved')

    def predict_param(self):
        dataset = Dataset(self.path2dataset, self.path2csv, train=0)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        predicted_arr = np.empty([len(data_loader.dataset), 1], dtype=float)

        with torch.no_grad():
            c = 1
            counter =0
            d=e=f=0
            d1 = e1 = f1 = 0
            to_stop = 200
            for batch_num, data in enumerate(data_loader):
                if batch_num % to_stop == 0 and batch_num > 0:
                    print('sample num: {}'.format(batch_num))
                    break
                spec = data[0].float().to(self.device)
                label = data[1]
                if self.vae:
                    _, vector = self.VAE(spec)
                # else:
                #     vector = self.Encoder(spec)
                #     print(vector)
                vector = vector.cpu().numpy()
                vector = vector.squeeze()
                # vector = convert_label4output(vector)
                # vector = np.around(vector, decimals=2)
                vector = F.softmax(torch.from_numpy(vector)).argmax().item()/4
                if vector == label[0].item():
                    counter += 1
                predicted_arr[c] = vector
                predicted_arr[c+1] = label[0]
                c += 2
        print('acc: %{}'.format(counter/to_stop))
        return predicted_arr


def main():
    path2dataset = ["/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0/",
                    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0.csv"]

    path2save = "/home/moshelaufer/PycharmProjects/VAE2/data/_osc2"
    path2encoder = "/home/moshelaufer/PycharmProjects/VAE2/data/_osc2/modelVAE_KL2.pth"
    path2vae = "/home/moshelaufer/PycharmProjects/VAE2/data/_osc2/modelVAE_KL2.pth"

    inference_model = Results(path2save, path2encoder, path2vae, path2dataset[0], path2dataset[1])
    inference_model.load_weight_model()

    predicted_arr = inference_model.predict_param()
    inference_model.save_results2csv(predicted_arr)
    print('Predicted process of VAE is done')



if __name__ == "__main__":
    main()
