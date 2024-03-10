import json


def load_kiwi_pos_dict() -> dict:
    with open('kiwi_pos_tags.json', 'r') as f:
        json_val = '\n'.join(map(lambda line: line.replace('\n', ''), f.readlines()))
        kiwi_pos_dict = json.loads(json_val)
        return kiwi_pos_dict
