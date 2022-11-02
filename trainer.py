import torch
import numpy as np
from model import model_checkpoints
from dataloader import dataloader
from transformers import AdamW
from torch.cuda import is_available
from tqdm import tqdm
from sklearn.metrics import classification_report
from utils import get_labels_dict


labels_dict = get_labels_dict()
device = 'cuda' if is_available() else 'cpu'
special_tokens = [
    "[UNK]",
    "[SEP]",
    "[PAD]",
    "[CLS]",
    "[MASK]"
]


def train_eval_ko_ner_model(model_checkpoint, num_epochs=5):

    def filter_special_tokens(b_input_ids, tensors):
        tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
        for input_ids, tensor in zip(b_input_ids, tensors):
            input_tokens = np.array(tokenizer.convert_ids_to_tokens(input_ids))
            filtered_tensor = tensor[
                (input_tokens != special_tokens[0]) &
                (input_tokens != special_tokens[1]) &
                (input_tokens != special_tokens[2]) &
                (input_tokens != special_tokens[3]) &
                (input_tokens != special_tokens[4])
                ]
            yield filtered_tensor

    train_dataloader = dataloader(is_train=True, device=device)
    eval_dataloader = dataloader(is_train=False, device=device)

    model_class, tokenizer_class = model_checkpoints.get(model_checkpoint)
    model = model_class.from_pretrained(model_checkpoint, num_labels=len(labels_dict))
    model.to(device)
    optim = AdamW(model.parameters(), lr=2e-5)

    # Training
    for epoch in range(num_epochs):
        train_loop = tqdm(train_dataloader, leave=True, desc=f'Epoch : {epoch}')
        total_loss = 0
        for batch in train_loop:
            optim.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optim.step()
            loss_val = round(loss.item(), 3)
            total_loss += loss_val
            train_loop.set_postfix(loss=loss_val)
        avg_loss = round(total_loss / len(train_dataloader), 3)
        checkpoint = f'{model.__class__.__name__}_epoch_{epoch}_avg_loss_{avg_loss}.pt'
        model.save_pretrained(checkpoint)

    # Evaluation
    with torch.no_grad():
        y_true, y_pred = [], []
        eval_loop = tqdm(eval_dataloader, leave=True, desc=f'Evaluation')
        for batch in eval_loop:
            b_input_ids = batch.get("input_ids")

            # (batch_size, sequence_length)
            labels = batch.get("labels")

            # logits : (batch_size, sequence_length, config.num_labels)
            # probs : (batch_size, sequence_length, config.num_labels)
            # result : (batch_size, sequence_length)
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            preds = filter_special_tokens(b_input_ids, preds)
            labels = filter_special_tokens(b_input_ids, labels)

            for pred, label in zip(preds, labels):
                y_true.extend(label.detach().cpu().numpy())
                y_pred.extend(pred.detach().cpu().numpy())

    report = classification_report(y_true=np.array(y_true),
                                   y_pred=np.array(y_pred))

    with open('report.txt', 'w') as f:
        print(report)
        f.write(report)


if __name__ == "__main__":
    train_eval_ko_ner_model("klue/bert-base")
