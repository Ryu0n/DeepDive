"""
Reference : https://aclanthology.org/W19-6120.pdf#page10
"""
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

model_name = 'bert-base-multilingual-cased'


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

    def get_tokens_in_sentence(self, sentence: str):
        tokenized_sentence = self.tokenizer.encode_plus(sentence)
        token_ids = tokenized_sentence.get('input_ids')
        tokens = [re.sub(' ', '', self.tokenizer.decode(token_id)) for token_id in token_ids]
        tokens = [token for token in tokens if token not in self.tokenizer.special_tokens_map.values()]
        return sentence, tokens

    def tokenize_by_pair(self, sentence: str, token: str):
        inputs = self.tokenizer.encode_plus(sentence, token,
                                            return_tensors='pt',
                                            max_length=512,
                                            padding=True,
                                            truncation=True)
        return inputs

    def forward(self, sentence: str):
        sentence, tokens = self.get_tokens_in_sentence(sentence)
        for token in tokens:
            inputs = self.tokenize_by_pair(sentence, token)
            latent_vec, _ = self.aspect_term_extractor(**inputs)
            is_entity = torch.argmax(latent_vec)


if __name__ == "__main__":
    sentence = '나는 사과를 먹었다.'
    model = CombinedModel()
    model(sentence)
