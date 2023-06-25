import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from attention import MultiHeadSelfAttention
from feedforward import FeedForwardNetwork


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int
    ):
        super(EncoderBlock).__init__()
        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads
        )
        self.layer_norm1 = nn.LayerNorm()
        self.feed_forward = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff
        )
        self.layer_norm2 = nn.LayerNorm()
    
    
    def forward(self, x, mask=None):
        _x = self.layer_norm1(
            self.self_attention(
                query=x,
                key=x,
                value=x,
                mask=mask
            )
        )
        x = _x + x  # residual connection
        
        _x = self.layer_norm2(
            self.feed_forward(
                x
            )
        )
        x = _x + x  # residual connection
        return x


class Encoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super(Encoder).__init__()
        
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
        
        self.encoder_blocks = [
            EncoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff
            ) 
            for _ in range(self.num_blocks)
        ]
    
    
    def forward(self, x):
        """
        x : (num_batch, seq_len)
        """
        
        embedding = self.embedding(x)  # (num_batch, seq_len, d_model)
        positional_encoding = self.positional_encoding(embedding)
        x = embedding + positional_encoding
        
        out = x
        for encoder_block in self.encoder_blocks:
            out = encoder_block(out)
        return out