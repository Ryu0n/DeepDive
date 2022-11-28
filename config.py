from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    num_epochs = 5
    batch_size = 8
    learning_rate = 2e-5
