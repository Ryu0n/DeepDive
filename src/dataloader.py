import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element


def _get_xml_file(file_name: str):
    cur_dir = os.path.abspath(os.curdir)
    return os.path.join(cur_dir, f'../data/{file_name}')


def _get_element_tree(file_name: str):
    xml_file = _get_xml_file(file_name)
    tree = ET.parse(xml_file)
    return tree


def parse_element_tree(file_name: str):
    tree = _get_element_tree(file_name)
    root = tree.getroot()
    for review in root:
        sentences = list(review)[0]
        for sentence in sentences:
            sentence = list(sentence)
            if len(sentence) != 2:
                continue
            text, opinions = sentence
            text = text.text
            print('\n', text)
            for opinion in opinions:
                opinion: Element
                polarity = opinion.get('polarity')
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                target = text[start:end] if start != 0 and end != 0 else opinion.get('target')
                print(target, polarity)


if __name__ == "__main__":
    parse_element_tree('ABSA16_Restaurants_Train_SB1_v2.xml')
