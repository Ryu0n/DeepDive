import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        max_length: int,
        d_model: int
    ):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(
            size=(
                max_length,
                d_model
            )
        )
        
        pos = torch.arange(
            start=0,
            end=max_length
        ).unsqueeze(dim=1)  # (max_len, 1)
        _2i = torch.arange(
            start=0,
            end=d_model,
            step=2
        )  # (d_model/2) = (256)
        
        self.encoding[:, 0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:, 1::2] = torch.cos(pos/(10000**(_2i/d_model)))
        
    
    def forward(self, embedding):
        _, seq_len, _ = embedding.shape
        return self.encoding[:seq_len, :]