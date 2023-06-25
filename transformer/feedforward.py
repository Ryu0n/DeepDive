import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int
    ):
        super(FeedForwardNetwork).__init__()
        self.d_model = d_model,
        self.d_ff = d_ff
        
        self.fc1 = nn.Linear(
            in_features=d_model,
            out_features=d_ff
        )
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(
            in_features=d_ff,
            out_features=d_model
        )
        
    
    def forward(self, x):
        return self.fc2(
            self.relu(
                self.fc1(
                    x
                )
            )
        )