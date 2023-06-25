import torch.nn as nn
from positional_encoding import PositionalEncoding
from attention import MultiHeadSelfAttention
from feedforward import FeedForwardNetwork


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int
    ):
        super(DecoderBlock).__init__()
        self.masked_self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads
        )
        self.layer_norm1 = nn.LayerNorm()
        self.cross_attention = None
        self.layer_norm2 = nn.LayerNorm()
        self.feed_forward = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff
        )
        self.layer_norm3 = nn.LayerNorm()
        
    
    def forward(self, x, mask=None):
        pass
        

class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super(Decoder).__init__()
        
        self.vocab_size = config.get("vocab_size")
        self.max_length = config.get("max_length")
        self.num_blocks = config.get("num_blocks")
        self.d_model    = config.get("d_model")
        self.num_heads  = config.get("num_heads")
        self.d_ff       = config.get("d_ff")
        
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model
        )
        self.positional_encoding = PositionalEncoding(
            max_length=self.max_length,
            d_model=self.d_model
        )
        self.decoder_blocks = [
            DecoderBlock()
            for _ in range(self.num_blocks)
        ]
        
    
    def forward(self):
        pass