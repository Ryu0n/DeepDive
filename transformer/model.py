import json
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(Transformer).__init__()
        with open("config.json", "r") as f:
            config = json.load(f)
            config_encoder: dict = config.get("encoder")
            config_decoder: dict = config.get("encoder")
            
            assert config_encoder is not None, "encoder configuration value is omitted."
            assert config_decoder is not None, "decoder configuration value is omitted."
            
            self.encoder = Encoder(config=config_encoder)
            self.decoder = Decoder(config=config_decoder)

    
    def encode(self, x):
        out = self.encoder(x)
        return out
        
    
    def decode(self, z, context):
        out = self.encoder(z, context)
        return out
    
    
    def forward(self, x, z):
        context = self.encode(x)
        return self.decode(z, context)