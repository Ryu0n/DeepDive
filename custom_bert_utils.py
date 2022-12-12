import json


def load_kiwi_pos_dict(noun: bool = False) -> dict:
    with open('kiwi_pos_tags.json', 'r') as f:
        json_val = '\n'.join(map(lambda line: line.replace('\n', ''), f.readlines()))
        kiwi_pos_dict = json.loads(json_val)
        if noun:
            tags = [k for k in kiwi_pos_dict.keys() if k.startswith('N')]
            return {k: i for i, k in enumerate(tags)}
        return kiwi_pos_dict
