import torch
import numpy as np
from tqdm import tqdm
from dataloader import dataloader
from torch.cuda import is_available
from torch.backends.mps import is_available as mps_is_available
from transformers import BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report
from config import Config as cfg


model_checkpoint = 'klue/bert-base'
device = 'cuda' if is_available() else 'cpu'
# device = 'mps' if mps_is_available() else 'cpu'  # for silicon mac


def train_sentimental_classifier(num_epochs=5, batch_size=16, learning_rate=2e-5):
    model = BertForSequenceClassification.from_pretrained(model_checkpoint)
    optim = AdamW(model.parameters(), lr=learning_rate)
    train_dataloader = dataloader(is_train=True, batch_size=batch_size)
    model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0
        train_dataloader = tqdm(train_dataloader, leave=True)

        for batch in train_dataloader:
            optim.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optim.step()
            loss_val = loss.item()
            total_loss += loss_val
            train_dataloader.set_postfix(loss=loss_val)

        avg_loss = round(total_loss / len(train_dataloader), 3)
        save_model_checkpoint = f'{model.__class__.__name__}_epoch_{epoch}_loss_{avg_loss}.pt'
        model.save_pretrained(save_model_checkpoint)

    return save_model_checkpoint


def evaluate_sentimental_classifier(save_model_checkpoint, batch_size=16):
    model = BertForSequenceClassification.from_pretrained(save_model_checkpoint)
    test_dataloader = tqdm(dataloader(is_train=False, batch_size=batch_size), leave=True)
    model.to(device)

    preds, labels = list(), list()
    for batch in test_dataloader:
        outputs = model(**batch)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1)
        pred = pred.detach().cpu().numpy()
        preds.append(pred)
        labels.append(
            batch.get('labels').detach().cpu().numpy()
        )
    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()
    report_text = classification_report(
        y_true=labels,
        y_pred=preds,
        target_names=['negative', 'positive']
    )
    with open('report.txt', 'w') as f:
        print(report_text)
        f.write(report_text)


if __name__ == "__main__":
    save_model_checkpoint = train_sentimental_classifier(num_epochs=cfg.num_epochs,
                                                         batch_size=cfg.batch_size,
                                                         learning_rate=cfg.learning_rate)
    evaluate_sentimental_classifier(save_model_checkpoint, batch_size=cfg.batch_size)
