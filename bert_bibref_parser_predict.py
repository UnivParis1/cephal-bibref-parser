#!/usr/bin/env python
# coding: utf-8
import argparse

from data import ModelWrapper
from install import install

DEFAULT_MAX_LENGTH = 128
MODEL_DIR = "./model"

install('torch')
install('transformers')
install('seqeval[gpu]')
install('sklearn')

from transformers import CamembertTokenizerFast, CamembertForTokenClassification
from utils import TextProcessor


def parse_arguments():
    parser = argparse.ArgumentParser(description='Fine tune Camembert for bibliographic references parsing.')
    parser.add_argument('--input', dest='input',
                        help='Input sentence', required=True)
    parser.add_argument('--model', dest='model',
                        help='Model path', default=MODEL_DIR)
    parser.add_argument('--max-length', dest='max_length',
                        help='Wordpiece tokenized sentence max length', default=DEFAULT_MAX_LENGTH)
    return parser.parse_args()


def main(arguments):
    sentence = arguments.input
    max_length = arguments.max_length
    model = arguments.model

    tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")
    bibref_parser_model = CamembertForTokenClassification.from_pretrained(model)

    wrapper = ModelWrapper(bibref_parser_model, tokenizer)

    sentence = TextProcessor.prepare(sentence)

    predictions_by_word = wrapper.predictions_by_word(sentence, max_length)
    print("```")
    print(sentence)
    print("```")
    print("```")
    print(list(zip(sentence.split(), predictions_by_word)))
    print("```")


if __name__ == '__main__':
    main(parse_arguments())
