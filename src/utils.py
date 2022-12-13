from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import ElectraTokenizerFast, ElectraForTokenClassification
from transformers import ElectraTokenizerFast, AutoModelForTokenClassification

from src.patterns import SingletonInstance

polarity_map = {
    'unrelated': 0,
    'negative': 1,
    'neutral': 2,
    'positive': 3,
}


PLM_CLASSES = {
    'klue/bert-base': [BertTokenizerFast, BertForTokenClassification],
    'bert-base-multilingual-cased': [BertTokenizerFast, BertForTokenClassification],
    'monologg/koelectra-base-v3-discriminator': [ElectraTokenizerFast, ElectraForTokenClassification],
    "beomi/KcELECTRA-base-v2022": [ElectraTokenizerFast, ElectraForTokenClassification]
}


class Arguments(SingletonInstance):
    def __init__(self, args):
        self.args = args
        self.args.train = True if self.args.train == 'True' else False
        self.args.eval = True if self.args.eval == 'True' else False
        self.model_path = self.args.model_path
        self.tokenizer_name = self.args.tokenizer_name
        self.tokenizer_class = PLM_CLASSES.get(self.tokenizer_name)[0]
        self.tokenizer = self.tokenizer_class.from_pretrained(self.tokenizer_name)
        self.model_class = PLM_CLASSES.get(self.model_path)[1]
