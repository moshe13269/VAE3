from VAE.Encoder import Encoder as Encoder
from VAE.VAE_model import VAE as VAE
import torch
import os.path


def initialize_encoder(Encoder, VAE):
    Encoder.conv1.weights = VAE.conv1.weights
    Encoder.conv2.weights = VAE.conv2.weights
    Encoder.conv3.weights = VAE.conv3.weights

    Encoder.conv4.weights = VAE.conv4.weights
    Encoder.conv5.weights = VAE.conv5.weights
    Encoder.conv6.weights = VAE.conv6.weights

    Encoder.conv7.weights = VAE.conv7.weights
    Encoder.conv8.weights = VAE.conv8.weights
    Encoder.conv9.weights = VAE.conv9.weights

    Encoder.conv10.weights = VAE.conv10.weights
    Encoder.conv11.weights = VAE.conv11.weights

    Encoder.conv12.weights = VAE.conv12.weights
    Encoder.conv13.weights = VAE.conv13.weights

    Encoder.conv14.weights = VAE.conv14.weights
    Encoder.conv15.weights = VAE.conv15.weights

    return Encoder


def load_models(path2vae):
    encoder = Encoder()
    vae = VAE()
    if os.path.isfile(path2vae):
        checkpoint = torch.load(path2vae)
        vae.load_state_dict(checkpoint['model_state_dict'])
    return initialize_encoder(encoder, vae)



