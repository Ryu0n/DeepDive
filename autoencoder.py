import numpy as npgit
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        word_vector_dim = 300
        self.encoder = nn.Sequential(
            nn.Linear(in_features=word_vector_dim, out_features=250),
            nn.LeakyReLU(),
            nn.Linear(in_features=250, out_features=200),
            nn.LeakyReLU(),
            nn.Linear(in_features=200, out_features=150),
            nn.LeakyReLU(),
            nn.Linear(in_features=150, out_features=100),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=100, out_features=150),
            nn.LeakyReLU(),
            nn.Linear(in_features=150, out_features=200),
            nn.LeakyReLU(),
            nn.Linear(in_features=200, out_features=250),
            nn.LeakyReLU(),
            nn.Linear(in_features=250, out_features=word_vector_dim)
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
    ae = AutoEncoder()
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


def compress_vector(ae: AutoEncoder, vectors: np.ndarray) -> np.ndarray:
    return ae.compress(torch.from_numpy(vectors).float()).detach().numpy()

