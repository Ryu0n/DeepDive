import torch
from custom_bert_utils import load_kiwi_pos_dict
from typing import List, Dict
from transformers import BertTokenizerFast
from kiwipiepy import Kiwi


def __tokenize_by_kiwi(sentences: List[str]) -> List[List[Dict]]:
    kiwi_tokenizer = Kiwi()
    kiwi_sentences_tokens = list(kiwi_tokenizer.tokenize(sentences))
    sentences_pos_spans = list()
    for kiwi_tokens in kiwi_sentences_tokens:
        pos_spans = list()
        for token in kiwi_tokens:
            pos_spans.append(
                {
                    'span': [token.start, token.end],
                    'pos': token.tag
                }
            )
        sentences_pos_spans.append(pos_spans.copy())
    return sentences_pos_spans


def __tokenize_by_bert(sentences: List[str], model_checkpoint: str):
    bert_tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
    inputs = bert_tokenizer(sentences,
                            return_offsets_mapping=True,
                            return_tensors='pt',
                            padding='max_length',
                            truncation=True)
    return bert_tokenizer, inputs


def tokenize(model_checkpoint: str, sentences: List[str]):
    """
    Custom Tokenization
    """
    """
    KIWI TOKENIZER
    """
    sentences_pos_spans = __tokenize_by_kiwi(sentences)

    """
    BERT TOKENIZER
    """
    bert_tokenizer, inputs = __tokenize_by_bert(sentences, model_checkpoint)
    # print(inputs)

    """
    POS TAGGING TO BERT TOKENS
    """
    sentences_input_ids = inputs.get('input_ids')
    token_to_id_vocab = bert_tokenizer.get_vocab()
    id_to_token_vocab = {v: k for k, v in token_to_id_vocab.items()}
    sentences_offsets = inputs.get('offset_mapping')
    special_tokens = bert_tokenizer.special_tokens_map.values()
    kiwi_pos_dict = load_kiwi_pos_dict()
    sentences_pos_tags = list()
    for sentence_index, (input_ids, offsets) in enumerate(zip(sentences_input_ids, sentences_offsets)):
        input_ids = input_ids.tolist()
        offsets = offsets.tolist()
        tokens = [(id_to_token_vocab.get(id), offset) for id, offset in zip(input_ids, offsets)]
        pos_tags = list()
        for token in tokens:
            if token[0] in special_tokens:
                pos_tags.append(kiwi_pos_dict.get("NOTHING"))
                continue
            bert_token_start, bert_token_end = token[-1]
            pos_spans = sentences_pos_spans[sentence_index]
            for pos_span in pos_spans:
                kiwi_start, kiwi_end = pos_span.get('span')
                if kiwi_start <= bert_token_start and bert_token_end <= kiwi_end:
                    pos_tags.append(
                        kiwi_pos_dict.get(
                            pos_span.get('pos')
                        )
                    )
        sentences_pos_tags.append(pos_tags.copy())
    sentences_pos_tags = torch.tensor(sentences_pos_tags, dtype=torch.int64)
    inputs['pos_tag_ids'] = sentences_pos_tags
    return inputs
