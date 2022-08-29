"""
Reference : https://aclanthology.org/W19-6120.pdf#page10
"""
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

model_name = 'bert-base-multilingual-cased'


def tokenize_sentence_pair(tokenizer, sentence: str, token: str):
    inputs = tokenizer.encode_plus(sentence, token,
                                   return_tensors='pt',
                                   max_length=512,
                                   padding=True,
                                   truncation=True)
    return inputs


class AspectTermExtractor(torch.nn.Module):
    """
    extract whether token is "related" or "unrelated".
    """

    def __init__(self):
        super().__init__()
        self.config = BertConfig.from_pretrained(model_name,
                                                 num_labels=2)  # related, unrelated
        self.bert = BertForSequenceClassification.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        logits, loss = outputs.logits, outputs.loss
        return logits, loss


class SentimentClassifier(torch.nn.Module):
    """
    extract sentimental expression of related aspect.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        pass


class CombinedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.aspect_term_extractor = AspectTermExtractor()
        self.sentiment_classifier = SentimentClassifier()

    def forward(self, sentence: str):
        tokenized_sentence = self.tokenizer.encode_plus(sentence)
        print(tokenized_sentence)
        token_ids = tokenized_sentence.get('input_ids')
        tokens = self.tokenizer.decode(token_ids)

        for token_id in token_ids:
            print(token_id, token_ids)

        print(self.tokenizer.get_vocab())

        print(tokens)

        # for token_id in token_ids:
        #     token = self.tokenizer.decode(token_id)
        #     token  = re.sub(' ', '', token)
        #     print(token_id, token, self.tokenizer(token))

        # for token_id in token_ids:
        #     token = self.tokenizer.decode(token_id)
        #     inputs = tokenize_sentence_pair(self.tokenizer, sentence, token)
            # print(sentence, token)
            # print(inputs, '\n')


if __name__ == "__main__":
    sentence = '나는 사과를 먹었다.'
    model = CombinedModel()
    model(sentence)
