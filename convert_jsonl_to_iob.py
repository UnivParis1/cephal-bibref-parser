from convert import convert_to_tags_format
import json
from transformers import pipeline


def to_json(annotations, classifier):
    data = {}
    counter = 0
    for annotation in annotations:
        paragraphs = annotation['paragraphs']
        if len(paragraphs) == 0:
            continue
        paragraph = paragraphs[0]
        title = paragraph['raw']
        language = classifier(title[0:512])[0]['label']
        print(f"{counter} {language}")
        tokens = paragraph['sentences'][0]['tokens']
        data.setdefault(language, [])
        data[language].append({'words': [token['orth'] for token in tokens],
                               'labels': [token['ner'] for token in tokens]})
        counter += 1
    return data


if __name__ == '__main__':
    classifier = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    annotations = convert_to_tags_format('auto.jsonl', format='IOB')
    json_data = to_json(annotations, classifier)
    for lng, value in json_data.items():
        with open(f"auto_tagged_bib_refs_{lng}.json", 'a') as file:
            json.dump(value, file)
