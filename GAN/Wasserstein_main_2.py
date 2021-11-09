import torch
import torch.optim as optim
from Decoder.Discriminator_2 import Discriminator as Discriminator
from Decoder.Generator_2 import Generator as Generator
from utils.dataloader import Dataset
from training_2 import Trainer


device = torch.device('cuda:2')
batch_size = 100

dataset = Dataset(
    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0/",
    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0.csv")
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                          pin_memory=True, drop_last=True)

generator = Generator()
generator.to(device)

discriminator = Discriminator()
discriminator.to(device)

generator.weight_init()
discriminator.weight_init()

# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# Train model
epochs = 100
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer, torch.device('cuda:2'))

trainer.train(data_loader, epochs)

# Save models
name = 'mnist_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')

# from torch.autograd import Variable
# import itertools


# def plot_output():
#     z_ = torch.randn((5 * 5, 100))  # .view(-1, 100, 1, 1)
#     z_ = Variable(z_.cuda(), volatile=True).to(device)
#
#     trainer.G.eval()
#     test_images = trainer.G(z_)
#     trainer.G.train()
#
#     grid_size = 5
#     fig, ax = plt.subplots(grid_size, grid_size, figsize=(5, 5))
#     for i, j in itertools.product(range(grid_size), range(grid_size)):
#         ax[i, j].get_xaxis().set_visible(False)
#         ax[i, j].get_yaxis().set_visible(False)
#     for k in range(grid_size * grid_size):
#         i = k // grid_size
#         j = k % grid_size
#         ax[i, j].cla()
#         ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(),
#                         cmap='gray')
#
#     plt.show()
#
#
# plot_output()