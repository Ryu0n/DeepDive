from src.patterns import SingletonInstance
from src.POSBert.custom_bert import CustomBertForTokenClassification
from src.POSBert.custom_bert_tokenizer import tokenize
from functools import partial
from transformers import BertTokenizerFast

polarity_map = {
    'unrelated': 0,
    'negative': 1,
    'neutral': 2,
    'positive': 3,
}


PLM_CLASSES = {
    'klue/bert-base': [BertTokenizerFast, CustomBertForTokenClassification, tokenize],
    'bert-base-multilingual-cased': [BertTokenizerFast, CustomBertForTokenClassification, tokenize],
}


class Arguments(SingletonInstance):
    def __init__(self, args):
        """
        Global args
        :param args:
        --train
        --eval
        --model_path
        --tokenizer_name
        """
        self.args = args
        self.args.train = True if self.args.train == 'True' else False
        self.args.eval = True if self.args.eval == 'True' else False
        self.model_path = self.args.model_path
        self.tokenizer_name = self.args.tokenizer_name
        self.tokenizer = PLM_CLASSES.get(self.tokenizer_name)[0].from_pretrained(self.tokenizer_name)
        self.tokenize_func = PLM_CLASSES.get(self.tokenizer_name)[2]
        self.tokenize_func = partial(self.tokenize_func, self.tokenizer_name)
        if 'bert' in self.args.tokenizer:
            self.model_class = PLM_CLASSES.get(self.tokenizer_name)[1]
