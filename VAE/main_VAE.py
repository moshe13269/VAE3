import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from VAE.VAE_model import VAE as VAE
from utils.dataloader import Dataset
import time
import os.path


def main():
    torch.cuda.empty_cache()
    file = open("/home/moshelaufer/PycharmProjects/VAE2/data/VAE/2_44/process_state_encoder_2.txt", "a")
    device = torch.device('cuda:3')
    model = VAE()

    path2model = "/home/moshelaufer/PycharmProjects/VAE2/data/VAE/2_44/model_encoder3_2.pt"\

    if os.path.isfile(path2model):
        checkpoint = torch.load(path2model)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.weight_init()
    model.to(device)

    model_optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    model.train()

    mse_criterion = nn.MSELoss().to(device)
    n_epochs = 100
    loss_list = []

    print('start epoch')
    file.write('start epoch\n')
    batch_size = 256

    for epoch in range(n_epochs):
        dataset = Dataset(
            "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0/",
            "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0.csv",
            model_type='vae')
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
            spec = spec.to(device)
            noise = torch.normal(mean=torch.zeros(spec.shape[0], 512, 4, 4)).to(device).requires_grad_(True)/4
            model_optimizer.zero_grad()
            specc_reconstructed = model(spec, noise)

            loss = mse_criterion(specc_reconstructed, spec)

            counter += 1
            loss.backward()
            model_optimizer.step()
            loss_tot += loss.item()

            if batch_num % 100 == 0 and batch_num > 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [Loss %f]VAE"
                    % (epoch, n_epochs, batch_num, num_batch, loss_tot/counter)
                )

        loss_tot = loss_tot / counter
        loss_list.append(loss_tot)

        file.write("--- %s seconds ---" % (time.time() - start_time))
        file.write('\n')
        file.write('loss_tot = %f ,VAE\n' % loss_tot)
        file.write("Loss tot= {}, epoch = {} wl".format(loss_tot, epoch))

        print("--- %s seconds ---" % (time.time() - start_time))
        print('\n')
        print("Loss train = {}, epoch = {}, batch_size = {} wl".format(loss_tot, epoch, batch_size))

        outfile_epoch = "/home/moshelaufer/PycharmProjects/VAE2/data/VAE/2_44/loss_arr_encoder3_22.npy"
        np.save(outfile_epoch, np.asarray(loss_tot))


        # need to edit
        if epoch <= 2:
            path = "/home/moshelaufer/PycharmProjects/VAE2/data/VAE/2_44/model_encoder3_22.pt"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model_optimizer.state_dict()}, path)
            print("Model had been saved")
        elif min(loss_list[:len(loss_list) - 2]) >= loss_list[len(loss_list) - 1]:
            path = "/home/moshelaufer/PycharmProjects/VAE2/data/VAE/2_44/model_encoder3_22.pt"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model_optimizer.state_dict()}, path)
            print("Model had been saved")

        if loss_tot <= 0.005:
            break;


    print("Training is over")
    file.write("Training is over\n")
    torch.no_grad()
    print("Weight file had successfully saved!!\n")
    file.close()


if __name__ == "__main__":
    main()
