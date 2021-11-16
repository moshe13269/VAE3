import sys
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.optim as optim
import os
import pickle
import timeit


class Trainer:
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer, device, gp_weight=10,
                 critic_iterations=5, print_every=150):
        self.device = device
        self.G = generator
        self.G.to(self.device)
        self.G_opt = optim.Adam(self.G.parameters(), lr=1e-4)
        self.D = discriminator
        self.D.to(self.device)
        self.D_opt = optim.Adam(self.D.parameters(), lr=1e-4)
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.losses_epochs = {'G': [0.0, 0], 'D': [0.0, 0], 'GP': [0.0, 0], 'gradient_norm': [0.0, 0]}
        self.num_steps = 0
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.epoch = 0
        self.path2save = '/home/moshelaufer/PycharmProjects/VAE2/data/GAN_noise_vector'
        self.file = open(os.path.join(self.path2save, 'results.txt'), 'a')

    def _critic_train_iteration(self, data):
        """ """
        batch_size = data[0].size()[0]
        generated_data = self.sample_generator(batch_size, data[1])

        if self.device:
            data = data[0].to(self.device)
        d_real = self.D(data, self.epoch)
        d_generated = self.D(generated_data, self.epoch)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses_epochs['GP'][0] += gradient_penalty.data.item()
        self.losses_epochs['GP'][1] += 1

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        self.losses_epochs['D'][0] += d_loss.data.item()
        self.losses_epochs['D'][1] += 1
        self.D_opt.step()

    def _generator_train_iteration(self, data):
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data[0].size()[0]
        generated_data = self.sample_generator(batch_size, data[1])

        # Calculate loss and optimize
        d_generated = self.D(generated_data, self.epoch)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses_epochs['G'][0] += g_loss.data.item()
        self.losses_epochs['G'][1] += 1

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data).to(self.device)
        if self.device:
            alpha = alpha.to(self.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated, self.epoch)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(self.device) if self.device else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses_epochs['gradient_norm'][0] += gradients.norm(2, dim=1).mean().data.item()
        self.losses_epochs['gradient_norm'][1] += 1

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def print2screen(self, i):
        try:
            print("Iteration {}".format(i + 1))
            print("D: {}".format(self.losses_epochs['D'][0] / self.losses_epochs['D'][1]))
            print("G: {}".format(self.losses_epochs['G'][0] / self.losses_epochs['G'][1]))
            print("GP: {}".format(self.losses_epochs['GP'][0] / self.losses_epochs['GP'][1]))
            print(
                "Gradient norm: {}".format(self.losses_epochs['gradient_norm'][0] / self.losses_epochs['gradient_norm'][1]))
            print('\n')
        except ZeroDivisionError:
            pass

    def print2txt(self, i):
        try:
            self.file.write("Iteration {}\n".format(i + 1))
            self.file.write("D: {}".format(self.losses_epochs['D'][0] / self.losses_epochs['D'][1]))
            self.file.write("G: {}".format(self.losses_epochs['G'][0] / self.losses_epochs['G'][1]))
            self.file.write("GP: {}".format(self.losses_epochs['GP'][0] / self.losses_epochs['GP'][1]))
            self.file.write("Gradient norm: {}".format(self.losses_epochs['gradient_norm'][0] /
                                                       self.losses_epochs['gradient_norm'][1]))
            self.file.write("\n")
        except ZeroDivisionError:
            pass

    def write2loss_list(self):
        self.losses['D'].append(self.losses_epochs['D'][0] / self.losses_epochs['D'][1])
        self.losses['G'].append(self.losses_epochs['G'][0] / self.losses_epochs['G'][1])
        self.losses['GP'].append(self.losses_epochs['GP'][0] / self.losses_epochs['GP'][1])
        self.losses['gradient_norm'].append(self.losses_epochs['gradient_norm'][0] /
                                            self.losses_epochs['gradient_norm'][1])

    def _train_epoch(self, data_loader):
        # start = timeit.timeit()
        start = timeit.default_timer()
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data)
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data)
            if i % self.print_every == 0:
                self.print2screen(i)
                self.print2txt(i)

        torch.save(
            {'epoch': len(self.losses['D']), 'D_state_dict': self.D.state_dict(), 'G_state_dict': self.G.state_dict()}
            , os.path.join(self.path2save, 'weight.pt'))

        self.write2loss_list()
        with open(os.path.join(self.path2save, 'losses.pickle'), 'wb') as handle:
            pickle.dump(self.losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model had been saved\n")
        # end = timeit.timeit()
        end = timeit.default_timer()
        print('time for epoch: {}'.format(end - start))
        self.file.write('time for epoch: {}\n\n'.format(end - start))

    def train(self, data_loader, epochs, save_training_gif=True):
        for epoch in range(epochs):
            self.epoch += 1
            print("\nEpoch {}".format(epoch + 1))
            self.file.write("Epoch {}\n".format(epoch + 1))
            self._train_epoch(data_loader)
            self.losses_epochs = {'G': [0.0, 0], 'D': [0.0, 0], 'GP': [0.0, 0], 'gradient_norm': [0.0, 0]}

    def sample_generator(self, num_samples, vector):
        generated_data = self.G(vector.to(self.device), self.epoch)
        return generated_data

