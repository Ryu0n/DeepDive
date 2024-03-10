import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            latent_dim: int,
            dropout_rate: float
        ):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=input_dim,
                out_features=hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        self.mu = nn.Linear(
            in_features=hidden_dim,
            out_features=latent_dim
        )
        self.logvar = nn.Linear(
            in_features=hidden_dim,
            out_features=latent_dim
        )
        
        
    def reparamerization(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)  # gaussian noise
        return mu + eps*std
        
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.fc2(
            self.fc1(
                x
            )
        )
        
        mu = F.relu(
            self.mu(
                x
            )
        )
        logvar = F.relu(
            self.logvar(
                x
            )
        )
        
        z = self.reparamerization(mu, logvar)
        return z, mu, logvar
