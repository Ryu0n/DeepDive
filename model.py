import json
import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification, AdamW
from PIL import Image
from datasets import load_metric


class SpamDataset(Dataset):
    def __init__(self, label_json: dict, train=True, train_ratio=0.7):
        """

        :param train: if train is True, use for train dataset
        :param train_ratio: train test ratio
        """
        self.contents = self.split_dataset_with_balancing(label_json, train, train_ratio)

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        return self.contents[index]

    def split_dataset_with_balancing(self, label_json, train, train_ratio):
        balance_dict = dict()
        for img_path, label in label_json.items():
            balance_dict.setdefault(label, list()).append(img_path)
        contents = list()
        for label, img_paths in balance_dict.items():
            split_index = int(len(img_paths) * train_ratio)
            if train:
                _img_paths = img_paths[:split_index]
            else:
                _img_paths = img_paths[split_index:]
            for img_path in _img_paths:
                contents.append((img_path, 1 if label == "1" else 0))
        return contents


def read_label_json():
    try:
        with open('label.json', 'r') as f:
            json_val = ''.join(f.readlines())
            return json.loads(json_val)
    except Exception as e:
        print(e)
        return dict()


if __name__ == "__main__":
    label_json = read_label_json()

    # Training
    train_dataset = SpamDataset(label_json)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model_checkpoint = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_checkpoint)
    model = ViTForImageClassification.from_pretrained(model_checkpoint)
    optim = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    model.train()
    num_epochs = 5
    for epoch in range(num_epochs):
        losses = []
        train_batches = tqdm.tqdm(train_dataloader, leave=True)
        for img_paths, labels in train_batches:
            optim.zero_grad()
            images = [Image.open(img_path) for img_path in img_paths]
            inputs = feature_extractor(images=images, do_resize=True, size=500, return_tensors="pt")
            outputs = model(**inputs)
            target = torch.LongTensor(labels)
            loss = criterion(outputs.logits, target)
            loss.backward()
            optim.step()
            loss_val = round(loss.item(), 3)
            losses.append(loss_val)
            train_batches.set_description(f'Epoch : {epoch}')
            loss_mean = round(sum(losses) / len(losses), 3)
            train_batches.set_postfix(loss=loss_mean)
        model_checkpoint = f'vit_epochs_{epoch}_loss_{loss_mean}.pt'
        model.save_pretrained(model_checkpoint)

    # Evaluation
    test_dataset = SpamDataset(label_json, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    test_batches = tqdm.tqdm(test_dataloader)
    fe_checkpoint = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTFeatureExtractor.from_pretrained(fe_checkpoint)
    model = ViTForImageClassification.from_pretrained(model_checkpoint)
    model.eval()
    true_labels, pred_labels = [], []
    for img_paths, labels in test_batches:
        images = [Image.open(img_path) for img_path in img_paths]
        inputs = feature_extractor(images=images, do_resize=True, size=500, return_tensors='pt')
        outputs = model(**inputs)
        preds = outputs.logits.argmax(-1)
        true_labels.append(labels)
        pred_labels.append(preds)
    true_labels, pred_labels = map(np.array, (true_labels, pred_labels))
    metric = load_metric("seqeval")
    result = metric.compute(predictions=pred_labels,
                            references=true_labels)
    print(result)
