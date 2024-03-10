from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    n_classes: int = 3
    latent_dim: int = 100
    ngf: int = 32
    ndf: int = 32
    channels: int = 3
    img_size: int = 256
    n_epochs: int = 50
    batch_size = 16
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999
    gen_dir: str = 'gen_images'
