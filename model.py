import json
import tqdm
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification, AdamW
from PIL import Image


class SpamDataset(Dataset):
    def __init__(self, label_json: dict):
        self.contents = [
            (np.array(Image.open(img_path)), 1 if label == "1" else 0)
            for img_path, label in label_json.items()
        ]

    def __len__(self):
        return len(self.contents)
    
    def __getitem__(self, index):
        return self.contents[index]


def read_label_json():
    try:
        with open('label.json', 'r') as f:
            json_val = ''.join(f.readlines())
            return json.loads(json_val)
    except:
        return dict()


if __name__ == "__main__":
    label_json = read_label_json()
    dataset = SpamDataset(label_json)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model_checkpoint = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_checkpoint)
    model = ViTForImageClassification.from_pretrained(model_checkpoint)
    optim = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    num_epochs = 5
    for epoch in range(num_epochs):
        losses = []
        batches = tqdm.tqdm(dataloader, leave=True)
        for img_path, label in batches:
            optim.zero_grad()
            image = np.array(Image.open(img_path))
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            target = torch.LongTensor([label])
            loss = criterion(outputs.logits, target)
            loss.backward()
            optim.step()
            loss_val = round(loss.item(), 3)
            losses.append(loss_val)
            batches.set_description(f'Epoch : {epoch}')
            loss_mean = round(sum(losses) / len(losses), 3)
            batches.set_postfix(loss=loss_mean)
        checkpoint = f'vit_epochs_{epoch}_loss_{loss_mean}.pt'
        model.save_pretrained(checkpoint)
