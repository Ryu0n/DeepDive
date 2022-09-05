from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import ElectraTokenizerFast, ElectraForTokenClassification

from src.patterns import SingletonInstance

polarity_map = {
    'positive': 3,
    'neutral': 2,
    'negative': 1,
    'unrelated': 0
}

is_entity = {
    'negative': 0,
    'positive': 1
}

PLM_CLASSES = {
    'bert-base-multilingual-cased': [BertTokenizerFast, BertForTokenClassification],
    'monologg/koelectra-base-v3-discriminator': [ElectraTokenizerFast, ElectraForTokenClassification]
}


class Arguments(SingletonInstance):
    def __init__(self, args):
        self.args = args
        self.args.train = True if self.args.train == 'True' else False
        self.args.eval = True if self.args.eval == 'True' else False
        self.tokenizer_class = PLM_CLASSES.get(self.args.tokenizer)[0]
        if 'bert' in self.args.tokenizer:
            self.model_class = BertForTokenClassification
        elif 'electra' in self.args.tokenizer:
            self.model_class = ElectraForTokenClassification
