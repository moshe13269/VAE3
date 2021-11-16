import torch
import torch.optim as optim
from Decoder.Discriminator_noise_vector import Discriminator as Discriminator
from Decoder.Generator_noise_vector import Generator as Generator
from utils.dataloader import Dataset
from training_noise_vector import Trainer


batch_size = 100

dataset = Dataset(
    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0/",
    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0.csv")
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                          pin_memory=True, drop_last=True)

device = torch.device('cuda:2')
generator = Generator()
generator.to(device)

discriminator = Discriminator()
discriminator.to(device)

generator.weight_init()
discriminator.weight_init()

lr = 1e-4
G_optimizer = optim.Adam(generator.parameters(), lr=lr)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# Train model
epochs = 100
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer, device)

trainer.train(data_loader, epochs)

# Save models
# torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
# torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
