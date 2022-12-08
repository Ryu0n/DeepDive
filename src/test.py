from transformers import BertForTokenClassification
from transformers import BertTokenizerFast


if __name__ == "__main__":
    model = BertForTokenClassification.from_pretrained('klue/bert-base')
    tokenizer = BertTokenizerFast.from_pretrained('klue/bert-base')
    print(model)
