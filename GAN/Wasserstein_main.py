import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataloaders import get_fashion_mnist_dataloaders
from Decoder.Discriminator import Discriminator
from Decoder.Generator import Generator
from utils.dataloader_gan import Dataset
from training import Trainer

cuda = torch.cuda.is_available()

device = torch.device('cuda')

data_loader, _ = get_fashion_mnist_dataloaders(batch_size=128)  # get_mnist_dataloaders(batch_size=64)
img_size = (32, 32, 1)

generator = Generator(img_size=img_size, latent_dim=100, dim=16)
generator = nn.DataParallel(generator)
generator.to(device)

discriminator = Discriminator(img_size=img_size, dim=16)
discriminator = nn.DataParallel(discriminator)
discriminator.to(device)

print(generator)
print(discriminator)

# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 200
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())

trainer.train(data_loader, epochs, save_training_gif=True)

# Save models
name = 'mnist_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')

from torch.autograd import Variable
import itertools


def plot_output():
    z_ = torch.randn((5 * 5, 100))  # .view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda(), volatile=True).to(device)

    trainer.G.eval()
    test_images = trainer.G(z_)
    trainer.G.train()

    grid_size = 5
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(5, 5))
    for i, j in itertools.product(range(grid_size), range(grid_size)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(grid_size * grid_size):
        i = k // grid_size
        j = k % grid_size
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(),
                        cmap='gray')

    plt.show()


plot_output()