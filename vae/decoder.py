import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            latent_dim: int,
            dropout_rate: float
        ):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=latent_dim,
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
        
        self.fc3 = nn.Linear(
            in_features=hidden_dim,
            out_features=input_dim
        )
        
    
    def forward(self, z):
        z = self.fc2(
            self.fc1(
                z
            )
        )
        
        # sigmoid = bernoulli distribution parameter (0 ~ 1)
        x_reconstruct = F.sigmoid(
            self.fc3(
                z
            )
        )
        return x_reconstruct