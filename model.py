import torch
import torch.nn as nn
from config import *


config = Config()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def conv_block(in_feat,
                       out_feat,
                       kernel_size,
                       stride,
                       padding,
                       normalize=True,
                       activation='leaky_relu'):
            layers = [
                nn.ConvTranspose2d(in_channels=in_feat,
                                   out_channels=out_feat,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   bias=False)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            activations = {'relu': nn.ReLU(True), 'leaky_relu': nn.LeakyReLU(0.01, inplace=True), 'tanh': nn.Tanh()}
            layers.append(activations[activation])
            return layers

        self.label_emb = nn.Embedding(config.n_classes, config.n_classes)
        self.upsampling = nn.Sequential(
            *conv_block(in_feat=config.latent_dim + config.n_classes,
                        out_feat=config.ngf * 32,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        activation='relu'),
            *conv_block(in_feat=config.ngf * 32,
                        out_feat=config.ngf * 16,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        activation='relu'),
            *conv_block(in_feat=config.ngf * 16,
                        out_feat=config.ngf * 8,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        activation='relu'),
            *conv_block(in_feat=config.ngf * 8,
                        out_feat=config.ngf * 4,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        activation='relu'),
            *conv_block(in_feat=config.ngf * 4,
                        out_feat=config.ngf * 2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        activation='relu'),
            *conv_block(in_feat=config.ngf * 2,
                        out_feat=config.ngf,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        activation='relu'),
            *conv_block(in_feat=config.ngf,
                        out_feat=config.channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        normalize=False,
                        activation='tanh'),
        )

    def forward(self, noise, labels):
        """

        :param noise: (batch_size, latent_dim)
        :param labels: (batch_size)
        :return:
        """
        # Concatenate label embedding and image to produce input
        # labels : (batch_size) -> (batch_size, n_classes)
        # gen_input : (batch_size, latent_dim + n_classes)
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        # gen_input : (batch_size, latent_dim + n_classes, 1, 1)
        for _ in range(2):
            gen_input = torch.unsqueeze(gen_input, -1)
        img = self.upsampling(gen_input)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(config.n_classes, config.img_size * config.img_size)

        def conv_block(in_feat,
                       out_feat,
                       kernel_size,
                       stride,
                       padding,
                       normalize=True,
                       activation='leaky_relu'):
            layers = [
                nn.Conv2d(in_channels=in_feat,
                          out_channels=out_feat,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=False)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            activations = {'leaky_relu': nn.LeakyReLU(0.2, inplace=True), 'tanh': nn.Tanh()}
            layers.append(activations[activation])
            return layers

        self.downsampling = nn.Sequential(
            *conv_block(in_feat=config.channels + 1,  # Channels + label_embed
                        out_feat=config.ndf,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            *conv_block(in_feat=config.ndf,
                        out_feat=config.ndf * 2,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            *conv_block(in_feat=config.ndf * 2,
                        out_feat=config.ndf * 4,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            *conv_block(in_feat=config.ndf * 4,
                        out_feat=config.ndf * 8,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            *conv_block(in_feat=config.ndf * 8,
                        out_feat=config.ndf * 16,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            *conv_block(in_feat=config.ndf * 16,
                        out_feat=config.ndf * 32,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            *conv_block(in_feat=config.ndf * 32,
                        out_feat=1,  # Fake or Real
                        kernel_size=4,
                        stride=1,
                        padding=0),  # torch.Size([64, 1, 1, 1])
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        """

        :param img: (batch_size, channels, img_size, img_size)
        :param labels: (batch_size)
        :return:
        """
        # Concatenate label embedding and image to produce input
        # label_embed : (batch_size, img_size^2) -> (batch_size, 1, img_size, img_size)
        label_embed = self.label_embedding(labels)
        label_embed = torch.reshape(label_embed, [-1, 1, config.img_size, config.img_size])
        # d_in : (batch_size, channels+1, img_size, img_size)
        d_in = torch.cat((img, label_embed), dim=1)
        validity = self.downsampling(d_in)
        return validity.squeeze()

