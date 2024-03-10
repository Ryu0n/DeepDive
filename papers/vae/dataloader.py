from torch.utils.data import DataLoader
from torchvision import datasets, transforms


transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

def load_dataloader(batch_size: int):
    train_dataset = datasets.MNIST(
        root="./data/",
        train=True,
        transform=transform,
        download=True
    )
    test_dataset = datasets.MNIST(
        root="./data/",
        train=False,
        transform=transform,
        download=True
    )
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True),  DataLoader(test_dataset, batch_size=batch_size, shuffle=True)