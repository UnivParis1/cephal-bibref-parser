from convert import convert_to_tags_format
import json
from transformers import pipeline


def to_json(annotations, classifier):
    data = {}
    for annotation in annotations:
        paragraphs = annotation['paragraphs']
        if len(paragraphs) == 0:
            continue
        paragraph = paragraphs[0]
        title = paragraph['raw']
        language = classifier(title)[0]['label']
        tokens = paragraph['sentences'][0]['tokens']
        data.setdefault(language, [])
        data[language].append({'words': [token['orth'] for token in tokens],
                     'labels': [token['ner'] for token in tokens]})
    return data


if __name__ == '__main__':
    classifier = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    annotations = convert_to_tags_format('all.jsonl', format='IOB')
    json_data = to_json(annotations, classifier)
    for lng, value in json_data.items():
        with open(f"tagged_bib_refs_{lng}.json", 'w') as file:
            json.dump(value, file)
