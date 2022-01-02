from VAE.Encoder import Encoder as Encoder
from VAE.VAE_model import VAE as VAE
import torch
import os.path


def initialize_encoder(Encoder, VAE):
    Encoder.conv1.weight.data = VAE.conv1.weight.data
    Encoder.conv2.weight.data = VAE.conv2.weight.data
    Encoder.conv3.weight.data = VAE.conv3.weight.data

    Encoder.conv4.weight.data = VAE.conv4.weight.data
    Encoder.conv5.weight.data = VAE.conv5.weight.data
    Encoder.conv6.weight.data = VAE.conv6.weight.data

    Encoder.conv7.weight.data = VAE.conv7.weight.data
    Encoder.conv8.weight.data = VAE.conv8.weight.data
    Encoder.conv9.weight.data = VAE.conv9.weight.data

    Encoder.conv10.weight.data = VAE.conv10.weight.data
    Encoder.conv11.weight.data = VAE.conv11.weight.data

    Encoder.conv12.weight.data = VAE.conv12.weight.data
    Encoder.conv13.weight.data = VAE.conv13.weight.data

    Encoder.conv14.weight.data = VAE.conv14.weight.data
    Encoder.conv15.weight.data = VAE.conv15.weight.data

    return Encoder


def load_models(path2vae):
    encoder = Encoder()
    vae = VAE()
    if os.path.isfile(path2vae):
        checkpoint = torch.load(path2vae)
        vae.load_state_dict(checkpoint['model_state_dict'])
    return initialize_encoder(encoder, vae)



