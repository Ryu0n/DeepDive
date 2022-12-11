import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class AutoEncoder(nn.Module):
    def __init__(self, vector_dim: int):
        super(AutoEncoder, self).__init__()
        self.vector_dim = vector_dim
        self.compressed_dim = self.vector_dim // 3
        compress_unit = (self.vector_dim - self.compressed_dim) // 4
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.vector_dim,
                      out_features=self.vector_dim - compress_unit),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.vector_dim - compress_unit,
                      out_features=self.vector_dim - 2*compress_unit),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.vector_dim - 2*compress_unit,
                      out_features=self.vector_dim - 3*compress_unit),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.vector_dim - 3*compress_unit,
                      out_features=self.compressed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.compressed_dim,
                      out_features=self.vector_dim - 3*compress_unit),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.vector_dim - 3*compress_unit,
                      out_features=self.vector_dim - 2*compress_unit),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.vector_dim - 2*compress_unit,
                      out_features=self.vector_dim - compress_unit),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.vector_dim - compress_unit,
                      out_features=self.vector_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x

    def compress(self, x):
        latent = self.encoder(x)
        return latent


class WordVectorDataset(Dataset):
    def __init__(self, sentence_vectors: np.ndarray):
        self.sentence_vectors = sentence_vectors

    def __len__(self):
        return len(self.sentence_vectors)

    def __getitem__(self, item):
        return torch.tensor(self.sentence_vectors[item], dtype=torch.float32)


def dataloader(sentence_vectors: np.ndarray):
    dataset = WordVectorDataset(sentence_vectors)
    return DataLoader(dataset, batch_size=2, shuffle=True)


def train_auto_encoder(sentence_vectors: np.ndarray, num_epochs=10, lr=0.005):
    ae = AutoEncoder(sentence_vectors.shape[-1])
    train_loader = dataloader(sentence_vectors)
    optim = torch.optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        train_loader = tqdm(train_loader, desc=f'Epoch : {epoch}')
        for vector in train_loader:
            pred = ae(vector)
            loss = criterion(vector, pred)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loader.set_postfix(loss=loss.item())

    return ae


def compress_vector(ae: AutoEncoder, sentence_vectors: np.ndarray) -> np.ndarray:
    return ae.compress(torch.from_numpy(sentence_vectors).float()).detach().numpy()

