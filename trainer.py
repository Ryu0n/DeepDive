import os
import torch
import numpy as np
from config import Config
from model import *
from tqdm import tqdm
from torch.cuda import is_available
from dataloader import load_dataloader
from torchvision.utils import save_image

config = Config()
FloatTensor = torch.cuda.FloatTensor if is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if is_available() else torch.LongTensor


def sample_image(generator, n_width, epoch, n_label=config.n_classes):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.Tensor(np.random.normal(0, 1, (n_width * n_label, config.latent_dim))).type(FloatTensor)
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([label for label in range(n_label) for _ in range(n_width)])
    labels = torch.Tensor(labels).type(LongTensor)
    gen_imgs = generator(z, labels)
    if not os.path.exists(config.gen_dir):
        os.mkdir(config.gen_dir)
    img_path = f"{config.gen_dir}/{epoch:06d}.png"
    save_image(gen_imgs.data, img_path, nrow=n_width, normalize=True)
    return img_path


def train_gan():
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=config.lr,
        betas=(config.b1, config.b2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.lr,
        betas=(config.b1, config.b2)
    )
    adversarial_loss = torch.nn.MSELoss()

    device = 'cuda' if is_available() else 'cpu'
    if is_available():
        generator.to(device)
        discriminator.to(device)
        adversarial_loss.to(device)

    train_dataloader, test_dataloder = load_dataloader()
    for epoch in range(config.n_epochs):
        dataloader = tqdm(train_dataloader, leave=True, desc=f'Epoch : {epoch}')
        for labels, images in dataloader:
            valid = FloatTensor(config.batch_size, 1).fill_(1.0)
            fake = FloatTensor(config.batch_size, 1).fill_(0.0)

            real_images = images.type(FloatTensor)
            labels = labels.type(LongTensor)

            """
            Generator Training
            """
            optimizer_G.zero_grad()

            # Generated Gaussian noise & labels
            z = FloatTensor(np.random.normal(size=(config.batch_size, config.latent_dim)))
            gen_labels = LongTensor(np.random.randint(0, config.n_classes, config.batch_size))

            # Generate images
            gen_images = generator(z, gen_labels)

            # Optimize generator to generate real images
            validity_real = discriminator(gen_images, gen_labels)
            g_loss = adversarial_loss(validity_real, valid)

            g_loss.backward()
            optimizer_G.step()

            """
            Discriminator Training
            """
            optimizer_D.zero_grad()

            # Optimize discriminator to discriminate real images
            validity_real = discriminator(real_images, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Optimize discriminator to discriminate fake images
            validity_fake = discriminator(gen_images.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss_val = round(d_loss.item(), 3)

            d_loss.backward()
            optimizer_D.step()

            dataloader.set_postfix(loss=d_loss_val)

        sample_image(generator, n_width=10, epoch=epoch)
        generator_checkpoint = f'generator_epoch_{epoch}_loss_{d_loss_val}.pt'
        discriminator_checkpoint = f'discriminator_epoch_{epoch}_loss_{d_loss_val}.pt'
        torch.save(generator.state_dict(), generator_checkpoint)
        torch.save(discriminator.state_dict(), discriminator_checkpoint)


def load_gan(generator_checkpoint: str, discriminator_checkpoint: str):
    generator = Generator().load_state_dict(torch.load(generator_checkpoint))
    discriminator = Discriminator().load_state_dict(torch.load(discriminator_checkpoint))


if __name__ == "__main__":
    train_gan()
