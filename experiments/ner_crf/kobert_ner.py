import glob
import os
import torch
import crf
import torchcrf

from utils.utils import MODEL_CLASSES


def get_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), 'r', encoding='utf-8')]


def parse_label(args):
    label_lst = get_labels(args)
    num_labels = len(label_lst)
    return label_lst, num_labels


def load_pretrained_kobert(args):
    args = args
    label_lst, num_labels = parse_label(args)
    config_class, model_class, _ = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task,
                                          id2label={str(i): label for i, label in enumerate(label_lst)},
                                          label2id={label: i for i, label in enumerate(label_lst)})
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    return model


class KoBERT_TorchCRF(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        label_lst, num_labels = parse_label(args)
        self.pad_token_id = label_lst.index('O')
        self.model = load_pretrained_kobert(args)
        self.dropout = torch.nn.Dropout()
        self.classifier = torch.nn.Linear(in_features=args.hidden_size,
                                          out_features=num_labels)
        self.crf_layer = torchcrf.CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        logits = outputs[0]
        logits = self.dropout(logits)
        logits = self.classifier(logits)

        loss = None
        if labels is not None:
            ignore_id = torch.nn.CrossEntropyLoss().ignore_index
            labels[labels == ignore_id] = self.pad_token_id
            log_likelihood, tags = self.crf_layer(logits, labels), self.crf_layer.decode(logits)
            loss = 0 - log_likelihood

        else:
            tags = self.crf_layer.decode(logits)
        tags = torch.Tensor(tags)
        return loss, tags


class KoBERT_CRF(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        label_lst, num_labels = parse_label(args)
        self.pad_token_id = label_lst.index('O')
        pad_token_label_id = label_lst.index('O')
        self.model = load_pretrained_kobert(args)
        self.dropout = torch.nn.Dropout()
        self.classifier = torch.nn.Linear(in_features=args.hidden_size,
                                          out_features=num_labels)
        self.crf_layer = crf.CRF(nb_labels=num_labels,
                                 bos_tag_id=crf.Const.BOS_TAG_ID,
                                 eos_tag_id=crf.Const.EOS_TAG_ID,
                                 pad_tag_id=pad_token_label_id,
                                 batch_first=True)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        logits = outputs[0]
        logits = self.dropout(logits)
        logits = self.classifier(logits)

        nll = None
        if labels is not None:
            ignore_id = torch.nn.CrossEntropyLoss().ignore_index
            labels[labels == ignore_id] = self.pad_token_id
            scores, tags = self.crf_layer.decode(logits)
            nll = self.crf_layer(logits, labels)

        else:
            tags = self.crf_layer.decode(logits)
        tags = torch.Tensor(tags)
        return nll, tags


class KoBERT_LSTM_TorchCRF(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        label_lst, num_labels = parse_label(args)
        self.pad_token_id = label_lst.index('O')
        self.model = load_pretrained_kobert(args)
        self.dropout = torch.nn.Dropout()
        self.lstm = torch.nn.LSTM(args.hidden_size, args.hidden_size // 2, batch_first=True)
        self.fc = torch.nn.Linear(in_features=args.hidden_size // 2,
                                  out_features=num_labels)
        self.crf_layer = torchcrf.CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        logits = outputs[0]
        logits = self.dropout(logits)
        logits, (h, c) = self.lstm(logits)
        logits = self.fc(logits)

        loss = None
        if labels is not None:
            ignore_id = torch.nn.CrossEntropyLoss().ignore_index
            labels[labels == ignore_id] = self.pad_token_id
            log_likelihood, tags = self.crf_layer(logits, labels), self.crf_layer.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf_layer.decode(logits)
        tags = torch.Tensor(tags)
        return loss, tags


class KoBERT_BiLSTM_TorchCRF(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        label_lst, num_labels = parse_label(args)
        self.pad_token_id = label_lst.index('O')
        self.model = load_pretrained_kobert(args)
        self.dropout = torch.nn.Dropout()
        self.lstm = torch.nn.LSTM(args.hidden_size, args.hidden_size // 2, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(in_features=args.hidden_size,
                                  out_features=num_labels)
        self.crf_layer = torchcrf.CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        logits = outputs[0]
        logits = self.dropout(logits)
        logits, (h, c) = self.lstm(logits)
        logits = self.fc(logits)

        loss = None
        if labels is not None:
            ignore_id = torch.nn.CrossEntropyLoss().ignore_index
            labels[labels == ignore_id] = self.pad_token_id
            log_likelihood, tags = self.crf_layer(logits, labels), self.crf_layer.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf_layer.decode(logits)
        tags = torch.Tensor(tags)
        return loss, tags


class KoELECTRA_TorchCRF(KoBERT_TorchCRF):
    def __init__(self, args):
        super().__init__(args)


class KoELECTRA_LSTM_TorchCRF(KoBERT_LSTM_TorchCRF):
    def __init__(self, args):
        super().__init__(args)


class KoELECTRA_BiLSTM_TorchCRF(KoBERT_BiLSTM_TorchCRF):
    def __init__(self, args):
        super().__init__(args)


def save_kobert_crf(args, model: torch.nn.Module, f1=None):
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_name = f'{model.__class__.__name__}_model'
    model_name = f'{model_name}_f1_{str(round(f1, 2))}' if f1 else model_name
    model_name = os.path.join(model_dir, f'{model_name}.pth')
    torch.save(model, model_name)


def load_kobert_crf(args):
    model_type = args.model_type
    model_dir = args.model_dir
    model_class_name = CUSTOM_MODEL_CLASSES.get(model_type).__name__
    prefix = os.path.join(model_dir, model_class_name)
    pths = sorted(glob.glob(f'{prefix}*.pth'), reverse=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(pths[0], map_location=device)
    print(model)
    return model


CUSTOM_MODEL_CLASSES = {
    'kobert-crf': KoBERT_CRF,
    'kobert-torchcrf': KoBERT_TorchCRF,
    'kobert-lstm-torchcrf': KoBERT_LSTM_TorchCRF,
    'kobert-bilstm-torchcrf': KoBERT_BiLSTM_TorchCRF,
    'koelectra-base-v3-torchcrf': KoELECTRA_TorchCRF,
    'koelectra-base-v3-lstm-torchcrf': KoELECTRA_LSTM_TorchCRF,
    'koelectra-base-v3-bilstm-torchcrf': KoELECTRA_BiLSTM_TorchCRF
}
