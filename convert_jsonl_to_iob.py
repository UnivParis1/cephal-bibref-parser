from convert import convert_to_tags_format
import json
from transformers import pipeline

INPUT_FILE = 'allhall.jsonl'


def to_json(counter, annotations, classifier):
    paragraphs = annotation['paragraphs']
    if len(paragraphs) == 0:
        return
    paragraph = paragraphs[0]
    title = paragraph['raw']
    language = classifier(title[0:512])[0]['label']
    print(f"{counter} {language}")
    tokens = paragraph['sentences'][0]['tokens']
    words = [token['orth'] for token in tokens]
    if ' '.join(words) == paragraph['raw']:
        value = {'words': words,
                 'labels': [token['ner'] for token in tokens]}
        with open(f"bio_tagged/all_tagged_bib_refs_{language}.json", 'a') as file:
            json.dump(value, file, indent=2)
    else:
        print("sentence integrity issue")
        print(' '.join(words))
        print(paragraph['raw'])


if __name__ == '__main__':
    classifier = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    for index, annotation in enumerate(convert_to_tags_format(INPUT_FILE, format='IOB')):
        to_json(index, annotation, classifier)
