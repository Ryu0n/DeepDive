from transformers import BertForSequenceClassification, BertTokenizer

model_checkpoint = 'monologg/kobert'
model = BertForSequenceClassification.from_pretrained(model_checkpoint)
tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
