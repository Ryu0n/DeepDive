import torch.nn as nn
from positional_encoding import PositionalEncoding
from attention import MultiHeadAttention
from feedforward import FeedForwardNetwork


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int
    ):
        super(DecoderBlock, self).__init__()
        self.masked_self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        
        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff
        )
        self.layer_norm3 = nn.LayerNorm(d_model)
        
    
    def forward(self, x, enc_out=None, self_mask=None, cross_mask=None):
        _x = self.layer_norm1(
            self.masked_self_attention(
                query=x,
                key=x,
                value=x,
                mask=self_mask
            )
        )
        x = _x + x  # residual connection
        
        if enc_out is not None:
            _x = self.layer_norm2(
                self.cross_attention(
                    query=x,
                    key=enc_out,
                    value=enc_out,
                    mask=cross_mask
                )
            )
            x = _x + x  # residual connection
        
        _x = self.layer_norm3(
            self.feed_forward(
                x
            )
        )
        x = _x + x  # residual connection
        return x
        

class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super(Decoder, self).__init__()
        
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
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.linear = nn.Linear(
            in_features=self.d_model,
            out_features=self.vocab_size
        )
        
    
    def forward(self, x, enc_out=None, dst_mask=None, src_dst_mask=None):
        """
        x : (num_batch, seq_len)
        """
        
        embedding = self.embedding(x)  # (num_batch, seq_len, d_model)
        positional_encoding = self.positional_encoding(embedding)
        x = embedding + positional_encoding
        
        out = x
        for decoder_block in self.decoder_blocks:
            out = decoder_block(
                out, 
                enc_out,
                dst_mask,
                src_dst_mask,
            )
        out = self.linear(out)
        return out