import json
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from mask import init_mask


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        with open("config.json", "r") as f:
            config = json.load(f)
            config_encoder: dict = config.get("encoder")
            config_decoder: dict = config.get("encoder")
            
            assert config_encoder is not None, "encoder configuration value is omitted."
            assert config_decoder is not None, "decoder configuration value is omitted."
            
            self.encoder = Encoder(config=config_encoder)
            self.decoder = Decoder(config=config_decoder)

    
    def encode(self, x, src_mask=None):
        context = self.encoder(x, src_mask)
        return context
        
    
    def decode(self, z, context, dst_mask=None, src_dst_mask=None):
        out = self.decoder(z, context, dst_mask, src_dst_mask)
        return out
    
    
    def forward(self, x, z):
        src_mask = init_mask(
            Q=x,
            K=x,
            method="padding"
        )
        dst_mask = init_mask(
            Q=z,
            K=z,
            method="tril"
        )
        src_dst_mask = init_mask(
            Q=z,
            K=x,
            method="padding"
        )
        context = self.encode(x, src_mask=src_mask)
        return self.decode(z, context, dst_mask=dst_mask, src_dst_mask=src_dst_mask)