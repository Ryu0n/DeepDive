from transformers import BertForTokenClassification, BertTokenizerFast


model_checkpoints = {
    'klue/bert-base': [BertForTokenClassification, BertTokenizerFast],
    'bert-base-multilingual-cased': [BertForTokenClassification, BertTokenizerFast]
}
