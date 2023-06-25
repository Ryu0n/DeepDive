import torch
import torch.nn as nn
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int
    ):
        super(MultiHeadAttention).__init__()
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        
        self.d_k: int = self.d_model // self.num_heads
        self.d_v: int = self.d_model // self.num_heads
        
        self.mat_query = nn.Linear(
            in_features=self.d_model, 
            out_features=self.d_model
        )
        self.mat_key   = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model
        )
        self.mat_value = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model
        )
        self.mat_out = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model
            
        )
        
    def calculate_attention(self, Q, K, V, mask=None):
        Q = rearrange(
            Q,
            "num_batch seq_len (num_heads d_k) -> num_batch num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k = self.d_k
        )
        K_T = rearrange(
            K,
            "num_batch seq_len (num_heads d_k) -> num_batch num_heads d_k seq_len",
            num_heads=self.num_heads,
            d_k = self.d_k
        )
        V = rearrange(
            V,
            "num_batch seq_len (num_heads d_v) -> num_batch num_heads seq_len d_v",
            num_heads=self.num_heads,
            d_k = self.d_v
        )
        energy = torch.matmul(Q, K_T) # (num_batch, num_heads, seq_len, seq_len)
        scaled_energy = energy / torch.sqrt(self.d_model)
        
        if mask is not None:
            scaled_energy = torch.masked_fill(
                input=scaled_energy,
                mask=(mask==0),
                value=-1e+4
            )
        
        attention_energy = torch.softmax(scaled_energy, dim=-1)
        attention_value = torch.matmul(attention_energy, V) # (num_batch, num_heads, seq_len, d_v)
        attention_value = rearrange(
            attention_value,
            "num_batch num_heads seq_len d_v -> num_batch seq_len (num_heads d_v)",
            num_heads=self.num_heads,
            d_k = self.d_v
        )  # (num_batch, seq_len, d_model)
        return self.mat_out(attention_value)
    
    
    def forward(self, query, key, value, mask=None):
        """
        query, key, value : (num_batch, seq_len, d_model)
        mask : (num_batch, 1, seq_len, seq_len)
        """
        Q, K, V = self.mat_query(query), self.mat_key(key), self.mat_value(value)
        return self.calculate_attention(
            Q=Q,
            K=K,
            V=V,
            mask=mask
        )
