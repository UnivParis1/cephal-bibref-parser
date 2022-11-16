import argparse

from convert import convert_to_tags_format
import json
from transformers import pipeline

DEFAULT_INPUT_FILE = 'input.jsonl'
DEFAULT_OUTPUT_DIR = 'bio_tagged'
DEFAULT_FILENAME_PREFIX = 'tagged_bib_refs'


def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert jsonl file to IOB format.')
    parser.add_argument('--input', dest='input',
                        help='Input sentence', default=DEFAULT_INPUT_FILE)
    parser.add_argument('--output-dir', dest='output_dir',
                        help='Output directory', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--output-filename', dest='output_filename_prefix',
                        help='Output file name prefix', default=DEFAULT_FILENAME_PREFIX)
    return parser.parse_args()


def to_json(counter, annotation, classifier, output_directory, filename_suffix):
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
        with open(f"{output_directory}/{filename_suffix}_{language}.json", 'a') as file:
            json.dump(value, file, indent=2)
    else:
        print("sentence integrity issue")
        print(' '.join(words))
        print(paragraph['raw'])


def main(arguments):
    input_file = arguments.input
    output_directory = arguments.output_dir
    output_filename_prefix = arguments.output_filename_prefix
    classifier = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    for index, annotation in enumerate(convert_to_tags_format(input_file, format='IOB')):
        to_json(index, annotation, classifier, output_directory, output_filename_prefix)


if __name__ == '__main__':
    main(parse_arguments())
