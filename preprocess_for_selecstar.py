import json
from tqdm import tqdm
from glob import glob
from transformers import ElectraTokenizerFast


def get_labels_dict():
    labels_dict = dict()
    with open('labels_selectstar.txt', 'r') as f:
        for i, label in enumerate(f.readlines()):
            labels_dict[label.replace('\n', '')] = i
    return labels_dict


def preprocess(tokenizer: ElectraTokenizerFast, json_dict: dict, labels_dict: dict):

    def join_sentence_with_labels(sentence: str, tags: list):
        tokenized_sentence = tokenizer.encode_plus(
            sentence,
            padding='max_length',
            return_offsets_mapping=True
        )
        labels = list()
        tokens = tokenized_sentence.tokens()
        offsets = tokenized_sentence.get('offset_mapping')
        for token, offset in zip(tokens, offsets):
            label = 'O'
            offset_start = offset[0]
            for tag in tags:
                start_index, end_index, tagged_label = tag
                if start_index <= offset_start <= end_index:
                    label = tagged_label
            label = labels_dict.get(label)
            if token == "[PAD]":
                label = -100
            labels.append(str(label))
        joined_line = f'{sentence}\t{" ".join(labels)}'
        return joined_line


    for data in tqdm(json_dict.get("data"), leave=True):
        sentence = data.get("sentence")
        labels = data.get("labels")
        tags = list()
        for label in labels:
            location = label.get("location")
            start_index = location[0]
            end_index = location[1]
            tagged_label = label.get("category")
            tags.append([start_index, end_index, tagged_label])
        joined_line = join_sentence_with_labels(sentence=sentence, tags=tags)
        yield joined_line
    

def main(prefix: str = 'final_json', train_ratio=0.7):
    joined_lines = list()
    labels_dict = get_labels_dict()
    tokenizer = ElectraTokenizerFast.from_pretrained("beomi/KcELECTRA-base-v2022")
    for file_path in glob(f'{prefix}*'):
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            json_dict = json.loads(''.join(f.readlines()))
            for joined_line in preprocess(tokenizer, json_dict, labels_dict):
                joined_lines.append(joined_line)
    num_train_samples = int(len(joined_lines) * train_ratio)
    with open('select_star_preprocess_train.txt', 'w') as f:
        f.write('\n'.join(joined_lines[:num_train_samples]))
    with open('select_star_preprocess_test.txt', 'w') as f:
        f.write('\n'.join(joined_lines[num_train_samples:]))


if __name__ == "__main__":
    main()
