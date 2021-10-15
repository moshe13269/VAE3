import numpy as np
import torch
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import os
import pickle


class Trainer:
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer, device, gp_weight=10,
                 critic_iterations=5, print_every=500):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.losses_epochs = {'G': 0.0, 'D': 0.0, 'GP': 0.0, 'gradient_norm': 0.0}
        self.losses_counter = {'G': 0, 'D': 0, 'GP': 0, 'gradient_norm': 0}
        self.num_steps = 0
        self.device = device
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.loss_gp = []
        self.loss_d = []
        self.G.to(self.device)
        self.D.to(self.device)
        self.epoch = 0
        self.path2save = '/home/moshelaufer/PycharmProjects/VAE2/data/GAN'

    def _critic_train_iteration(self, data):
        """ """
        # Get generated data
        batch_size = data[0].size()[0]
        generated_data = self.sample_generator(batch_size, data[1])

        # Calculate probabilities on real and generated data
        data = Variable(data[0])
        if self.device:
            data = data.to(self.device)
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.data)  # data[0]##############
        self.losses_counter['GP'] += 1

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.D_opt.step()

        # Record loss
        self.losses_epochs['D'] += d_loss.data  # data[0]

    def _generator_train_iteration(self, data):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data[0].size()[0]
        generated_data = self.sample_generator(batch_size, data[1])

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.data)  # data[0]##############
        self.losses_counter['G'] += 1

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.device:
            alpha = alpha.to(self.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.device:
            interpolated = interpolated.to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(self.device) if self.device else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data)
        self.losses_counter['gradient_norm'] += 1

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        torch.save(
            {'epoch': len(self.losses['D']), 'D_state_dict': self.D.state_dict(), 'G_state_dict': self.G.state_dict(),
             'optimizer_state_dict_D': self.D_opt.state_dict(), 'optimizer_state_dict_G': self.G_opt.state_dict()}
            , os.path.join(self.path2save, 'weight.pt'))
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data)
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data)

            if i % self.print_every == 0 and len(self.losses['D']) > 0 and len(self.losses['GP']) > 0 and \
                    len(self.losses['gradient_norm']) > 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if self.num_steps > self.critic_iterations and len(self.losses['G']) > 0:
                    print("G: {}".format(self.losses['G'][-1]))

        if i % self.print_every == 0 and len(self.losses['D']) > 0 and len(self.losses['GP']) > 0 and \
                len(self.losses['gradient_norm']) > 0:
            self.loss_gp.append(self.losses['GP'][-1])
            self.loss_d.append(self.losses['D'][-1])
        torch.save(
            {'epoch': len(self.losses['D']), 'D_state_dict': self.D.state_dict(), 'G_state_dict': self.G.state_dict(),
             'optimizer_state_dict_D': self.D_opt.state_dict(), 'optimizer_state_dict_G': self.G_opt.state_dict()}
            , os.path.join(self.path2save, 'weight.pt'))

        with open(os.path.join(self.path2save, 'losses.pickle'), 'wb') as handle:
            pickle.dump(self.losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model had been saved")

    def losses_calculation(self):
        for keys in self.losses_epochs.keys():
            self.losses[keys].append(self.losses_epochs[keys] / max(self.losses_counter[keys], 1.0))

    def train(self, data_loader, epochs, save_training_gif=True):
        for epoch in range(epochs):
            self.epoch += 1
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)
            self.losses_epochs = {'G': 0.0, 'D': 0.0, 'GP': 0.0, 'gradient_norm': 0.0}
            self.losses_counter = {'G': 0, 'D': 0, 'GP': 0, 'gradient_norm': 0}

    def sample_generator(self, num_samples, vector):
        # latent_samples = Variable(self.G.module.sample_latent(num_samples))
        # if self.device:
        #     latent_samples = latent_samples.to(self.device)
        generated_data = self.G(vector.to(self.device), torch.ones((vector.shape[0], 512, 4, 4)).to(self.device))
        return generated_data

    # def sample(self, num_samples):
    #     generated_data = self.sample_generator(num_samples)
    #     # Remove color channel
    #     return generated_data.data.cpu().numpy()[:, 0, :, :]
