import pandas as pd
import torch
# from models.model_VAE_lfo1destination2 import VAE_lfo1destination2
from models.model_Encoder import Encoder
from utils.dataloader import Dataset
import numpy as np
import os
from utils.util import denormalized_vector, convert_label4output
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Results:

    def __init__(self, path2save, path2encoder, path2vae, path2dataset, path2csv, vae=0):
        self.path2save = path2save
        self.device = torch.device('cuda:3')
        self.path2Encoder = path2encoder
        self.path2VAE = path2vae
        self.Encoder = Encoder().float().to(self.device)
        # self.VAE = VAE_lfo1destination2().float().to(self.device)
        self.path2csv = path2csv
        self.path2dataset = path2dataset
        self.vae = vae

    def load_weight_model(self):
        self.Encoder.load_state_dict(torch.load(self.path2Encoder)['model_state_dict'])
        self.Encoder.eval()
        # self.VAE.load_state_dict(torch.load(self.path2VAE)['model_state_dict'])
        # self.VAE.eval()

    def save_results2csv(self, np_array):
        columns = ['24.osc2waveform', '26.lfo1waveform', '32.lfo1destination',
                   '30.lfo1amount', '28.lfo1rate', '3.cufoff'] #, '4.resonance'
        df = pd.DataFrame(np_array, columns=columns)
        if self.vae:
            df.to_csv(os.path.join(self.path2save, 'vae.csv'))
        else:
            df.to_csv(os.path.join(self.path2save, 'encoder.csv'))
        print('csv file had been saved')

    @staticmethod
    def convert_vector_to_label(vector, label):
        a0 = F.softmax(vector[:, :4].squeeze()).argmax().item()
        a1 = F.softmax(vector[:, 4:8].squeeze()).argmax().item()
        a2 = F.softmax(vector[:, 8:10].squeeze()).argmax().item()
        a3_5 = vector[:, 10:].cpu().numpy().squeeze()
        pred_label = np.concatenate((np.asarray([a0, a1, a2]), a3_5))
        return pred_label

    def predict_param(self):
        dataset = Dataset(self.path2dataset, self.path2csv, train=1)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        predicted_arr = np.empty([len(data_loader.dataset), 6], dtype=float)

        with torch.no_grad():
            c = 1
            counter =0
            d=e=f=0
            d1 = e1 = f1 = 0
            to_stop = 1000
            l = [0,0,0,0]
            for batch_num, data in enumerate(data_loader):
                if batch_num % to_stop == 0 and batch_num > 0:
                    print('sample num: {}'.format(batch_num))
                    break
                spec = data[0].float().to(self.device)
                label = data[1]
                if self.vae:
                    # _, vector = self.VAE(spec)
                    vector = self.VAE(spec)
                else:
                    vector = self.Encoder(spec)
                #     print(vector)
                # vector = vector.cpu().numpy()
                # vector = vector.squeeze()
                # # vector = convert_label4output(vector)
                # # vector = np.around(vector, decimals=2)
                # vector = F.softmax(torch.from_numpy(vector)).argmax().item()
                # if vector == int(label[0][0].item()):
                #     counter += 1
                # # print(label[0][2].item(), vector)
                # l[int(label[0][2].item())] += 1
                # if int(label[0][2].item()) != 0:
                #     d += 1
                pred_label = Results.convert_vector_to_label(vector, label.to(self.device))
                predicted_arr[c] = pred_label
                predicted_arr[c+1] = label.squeeze()
                predicted_arr[c+2] = np.asarray([0, 0, 0, 0, 0, 0])
                c += 3
        print('acc: %{}, d(0) %{}'.format(counter/to_stop, d/to_stop))
        print(l)
        return predicted_arr


def main():
    path2dataset = ["/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0/",
                    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0.csv"]

    path2save = "/home/moshelaufer/PycharmProjects/VAE2/data/encoder"
    path2encoder = "/home/moshelaufer/PycharmProjects/VAE2/data/encoder/model_encoder3.pt"
    path2vae = "/home/moshelaufer/PycharmProjects/VAE2/data/encoder/model_encoder3.pt"

    inference_model = Results(path2save, path2encoder, path2vae, path2dataset[0], path2dataset[1])
    inference_model.load_weight_model()

    predicted_arr = inference_model.predict_param()
    inference_model.save_results2csv(predicted_arr)
    print('Predicted process of VAE is done')



if __name__ == "__main__":
    main()
