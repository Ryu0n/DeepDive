import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self) -> None:
        super(EncoderBlock).__init__()
        self.self_attention = None
        self.feed_forward = None
    
    
    def forward(self, x):
        return self.feed_forward(
            self.self_attention(
                x
            )
        )


class Encoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super(Encoder).__init__()
        self.encoder_blocks = [
            EncoderBlock() for _ in range(config.get("num_blocks"))
        ]
        
        
    def positional_encoding(self, x):
        pass
    
    
    def forward(self, x):
        out = x
        for encoder_block in self.encoder_blocks:
            out = encoder_block(out)
        return out