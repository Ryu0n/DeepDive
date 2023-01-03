from transformers import BertForTokenClassification, BertTokenizerFast
from transformers import ElectraForTokenClassification, ElectraTokenizerFast


model_checkpoints = {
    'klue/bert-base': [BertForTokenClassification, BertTokenizerFast],
    'bert-base-multilingual-cased': [BertForTokenClassification, BertTokenizerFast],
    'monologg/kobert': [BertForTokenClassification, BertTokenizerFast],
    "beomi/KcELECTRA-base-v2022": [ElectraForTokenClassification, ElectraTokenizerFast]
}
