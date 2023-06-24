import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super(Decoder).__init__()