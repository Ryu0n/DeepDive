import gc
import torch
import numpy as np
from model import model_checkpoints
from dataloader import dataloader
from transformers import AdamW
from torch.cuda import is_available
from tqdm import tqdm
from sklearn.metrics import classification_report
from preprocess_for_selecstar import get_labels_dict
# from preprocess_for_kooklib import get_labels_dict  # 국립국어원 라벨


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

    model_class, tokenizer_class = model_checkpoints.get(model_checkpoint)
    model = model_class.from_pretrained(model_checkpoint, num_labels=len(labels_dict))
    tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
    model.to(device)
    optim = AdamW(model.parameters(), lr=2e-5)
    train_dataloader = dataloader(is_train=True, tokenizer=tokenizer, device=device)
    eval_dataloader = dataloader(is_train=False, tokenizer=tokenizer, device=device)

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
        checkpoint = f'ner_{model.__class__.__name__}_epoch_{epoch}_avg_loss_{avg_loss}.pt'
        model.save_pretrained(checkpoint)

    # Evaluation
    with torch.no_grad():
        y_true, y_pred = [], []
        eval_loop = tqdm(eval_dataloader, leave=True, desc=f'Evaluation')
        for batch in eval_loop:
            b_input_ids = batch.get("input_ids")
            labels = batch.get("labels")
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


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    clear_gpu_memory()
    train_eval_ko_ner_model("beomi/KcELECTRA-base-v2022")
