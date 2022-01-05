import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from VAE.fine_tuning_model import FineTuning as FineTuning
from utils.dataloader import Dataset
import time
from torch.nn import functional as F
from VAE.craete_encoder import load_models


def main():
    torch.cuda.empty_cache()
    file = open("/home/moshelaufer/PycharmProjects/VAE2/data/VAE/fine_tuning/process_state_encoder_2.txt", "a")
    device = torch.device('cuda:2')

    path2vae = "/home/moshelaufer/PycharmProjects/VAE2/data/VAE/2_44/model_encoder3_2.pt"
    encoder = load_models(path2vae)
    encoder.to(device)

    model_optimizer = optim.Adam(encoder.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min', verbose=True, min_lr=0.00001)
    encoder.train()

    mse_criterion = nn.MSELoss().to(device)
    ce_criterion = nn.CrossEntropyLoss().to(device)

    n_epochs = 50
    loss_list = []

    print('start epoch')
    file.write('start epoch\n')
    batch_size = 256

    for epoch in range(n_epochs):
        dataset = Dataset(
            "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0/",
            "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0.csv",
            model_type='vae_fine_tuning', train=0)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                                  pin_memory=True, drop_last=True)
        print(len(data_loader.dataset))

        num_batch = len(data_loader.dataset) // batch_size
        loss_tot = 0.0
        start_time = time.time()

        counter = 0
        for batch_num, data in enumerate(data_loader):
            if batch_num % 200 == 0:
                print("sum samples = {} ".format(batch_num * batch_size))
            spec = data[0].float()
            label = data[1].to(device)
            spec = spec.to(device)
            model_optimizer.zero_grad()
            latent_vector = encoder(spec)

            loss = ce_criterion(latent_vector[:, :4], label[:, :1].squeeze().long()) + \
                   ce_criterion(latent_vector[:, 4:8], label[:, 1:2].squeeze().long()) + \
                   ce_criterion(latent_vector[:, 8:10], label[:, 2:3].squeeze().long()) + \
                   mse_criterion(F.relu(latent_vector[:, 10:]), label[:, 3:])

            counter += 1
            loss.backward()
            model_optimizer.step()
            loss_tot += loss.item()

            if batch_num % 100 == 0 and batch_num > 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [Loss %f] fine_tune"
                    % (epoch, n_epochs, batch_num, num_batch, loss_tot / counter)
                )
            if epoch == 9:
                a = 0
        loss_tot = loss_tot / counter
        scheduler.step(loss_tot)
        loss_list.append(loss_tot)

        file.write("--- %s seconds ---" % (time.time() - start_time))
        file.write('\n')
        file.write('loss_tot = %f ,VAE\n' % loss_tot)
        file.write("Loss tot= {}, epoch = {} wl".format(loss_tot, epoch))

        print("--- %s seconds ---" % (time.time() - start_time))
        print('\n')
        print("Loss train = {}, epoch = {}, batch_size = {} wl".format(loss_tot, epoch, batch_size))

        outfile_epoch = "/home/moshelaufer/PycharmProjects/VAE2/data/VAE/fine_tuning/loss_arr_fine_tune.npy"
        np.save(outfile_epoch, np.asarray(loss_tot))

        # need to edit
        if epoch <= 2:
            path = "/home/moshelaufer/PycharmProjects/VAE2/data/VAE/fine_tuning/model_fine_tune.pt"
            torch.save({'epoch': epoch, 'model_state_dict': encoder.state_dict(),
                        'optimizer_state_dict': model_optimizer.state_dict()}, path)
            print("Model had been saved")
        elif min(loss_list[:len(loss_list) - 2]) >= loss_list[len(loss_list) - 1]:
            path = "/home/moshelaufer/PycharmProjects/VAE2/data/VAE/fine_tuning/model_fine_tune.pt"
            torch.save({'epoch': epoch, 'model_state_dict': encoder.state_dict(),
                        'optimizer_state_dict': model_optimizer.state_dict()}, path)
            print("Model had been saved")

    print("Training is over")
    file.write("Training is over\n")
    torch.no_grad()
    print("Weight file had successfully saved!!\n")
    file.close()


if __name__ == "__main__":
    main()
