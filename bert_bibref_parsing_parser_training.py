#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import itertools
import os

from install import install

install('numpy')
install('pandas')
install('torch')
install('transformers')
install('seqeval[gpu]')
install('sklearn')

import numpy as np
import pandas as pd
import torch
from torch import cuda
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report
from data import BibRefParserDataset, ModelWrapper

DEFAULT_SOURCE_DATA_FILE_NAME = 'iob.json'

DEFAULT_OUTPUT_DIR = "output"
DEFAULT_LANGUAGE = "fr"
MODEL_DIR = "./[OUTPUT]/[SUFFIX]"

DEFAULT_TRAINING_EPOCHS = 10
DEFAULT_TRAINING_BATCH_SIZE = 4
DEFAULT_VALIDATION_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 5e-06
DEFAULT_VALIDATION_SIZE = 0.1
DEFAULT_LOG_FREQUENCY = 100


def parse_arguments():
    parser = argparse.ArgumentParser(description='Fine tune Bert or Camembert for bibliographic references parsing.')
    parser.add_argument('--lng', dest='language', choices=['fr', 'en', 'multi'],
                        help='Expected language for titles', default=DEFAULT_LANGUAGE)
    parser.add_argument('--output', dest='output',
                        help='Output directory path', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--source', dest='source',
                        help='IOB source file', default=DEFAULT_SOURCE_DATA_FILE_NAME)
    parser.add_argument('--epochs', dest='epochs',
                        help='Number of training epochs', default=DEFAULT_TRAINING_EPOCHS)
    parser.add_argument('--max-length', dest='max_length',
                        help='Wordpiece tokenized sentence max length', default=ModelWrapper.DEFAULT_MAX_LENGTH)
    parser.add_argument('--learning-rate', dest='learning_rate',
                        help='Learning rate', default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--training-batch-size', dest='training_batch_size',
                        help='Training batch size', default=DEFAULT_TRAINING_BATCH_SIZE)
    parser.add_argument('--validation-batch-size', dest='validation_batch_size',
                        help='Validation batch size', default=DEFAULT_VALIDATION_BATCH_SIZE)
    parser.add_argument('--validation-size', dest='validation_size',
                        help='Validation dataset size', default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument('--log-frequency', dest='log_frequency',
                        help='Frequency of logs during training (number of batches)', default=DEFAULT_LOG_FREQUENCY)
    parser.add_argument('--limit', dest='limit',
                        help='Number of samples to take from source file', default=None)
    return parser.parse_args()


def extract_labels_from_data(data):
    unique_labels = np.unique(list(itertools.chain(*data.labels)))
    labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
    ids_to_labels = {v: k for v, k in enumerate(unique_labels)}
    print("Labels to ids :")
    print(labels_to_ids)
    num_labels = len(unique_labels)
    print(f"Number of labels : {num_labels}")
    return ids_to_labels, labels_to_ids, num_labels, unique_labels


def load_data(source_data_file_name, limit):
    data = pd.read_json(source_data_file_name)
    if limit:
        data = data[0:int(limit)]
    print("2 first lines of data :")
    print(data.head(2))
    print(f"Shape of data : {data.shape}")
    return data


def create_data_loaders(data, ids_to_labels, labels_to_ids, max_length, tokenizer, training_batch_size,
                        validation_batch_size, validation_size):
    training_dataset, testing_dataset = train_test_split(data, test_size=validation_size)
    training_dataset = training_dataset.reset_index(drop=True)
    testing_dataset = testing_dataset.reset_index(drop=True)
    training_torch_dataset = BibRefParserDataset(training_dataset, tokenizer, labels_to_ids, ids_to_labels,
                                                 max_length)
    validation_torch_dataset = BibRefParserDataset(testing_dataset, tokenizer, labels_to_ids, ids_to_labels,
                                                   max_length)
    training_torch_dataloader = DataLoader(training_torch_dataset, **{'batch_size': training_batch_size,
                                                                      'shuffle': True,
                                                                      'num_workers': 0
                                                                      })
    validation_torch_dataloader = DataLoader(validation_torch_dataset, **{'batch_size': validation_batch_size,
                                                                          'shuffle': True,
                                                                          'num_workers': 0
                                                                          })
    return training_torch_dataloader, validation_torch_dataloader


def main(arguments: argparse.Namespace) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    language = arguments.language
    print(f"Expected language for titles : {language}")
    output_dir = arguments.output
    print(f"Output directory : {output_dir}")
    source_data_file_name = arguments.source
    print(f"Source data file : {source_data_file_name}")
    epochs = int(arguments.epochs)
    print(f"Number of epochs : {epochs}")
    max_length = int(arguments.max_length)
    print(f"Tokenized sentence max length : {max_length}")
    learning_rate = float(arguments.learning_rate)
    print(f"Learning rate : {learning_rate}")
    training_batch_size = int(arguments.training_batch_size)
    print(f"Training batch size : {training_batch_size}")
    validation_batch_size = int(arguments.validation_batch_size)
    print(f"Validation batch size : {validation_batch_size}")
    validation_size = float(arguments.validation_size)
    print(f"Validation size : {validation_size}")
    limit = None if arguments.limit is None else int(arguments.limit)
    print(f"Max number of samples : {limit or 'no limit'}")
    log_frequency = int(arguments.log_frequency)
    print(f"Frequency of logs during training (number of batches) : {log_frequency}")

    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    e = datetime.datetime.now()
    marker = "%s-%s-%s-%s-%s-%s" % (e.year, e.month, e.day, e.hour, e.minute, e.second)
    print(f"Output folder marker : {marker}")

    data = load_data(source_data_file_name, limit)

    ids_to_labels, labels_to_ids, num_labels, unique_labels = extract_labels_from_data(data)

    tokenizer = AutoTokenizer.from_pretrained(get_config(language=language), use_fast=True)

    config = AutoConfig.from_pretrained(
        get_config(language=language),
        num_labels=num_labels,
        id2label=ids_to_labels,
        label2id=labels_to_ids)
    bibref_parser_model = AutoModelForTokenClassification.from_config(config)
    bibref_parser_model.to(device)

    training_torch_dataloader, validation_torch_dataloader = create_data_loaders(data, ids_to_labels, labels_to_ids,
                                                                                 max_length, tokenizer,
                                                                                 training_batch_size,
                                                                                 validation_batch_size, validation_size)
    optimizer = torch.optim.Adam(params=bibref_parser_model.parameters(), lr=learning_rate)

    wrapper = ModelWrapper(bibref_parser_model, tokenizer, device, optimizer)

    for epoch in range(epochs):
        print(f"Epoch n°: {epoch + 1}")
        wrapper.train_model(training_torch_dataloader, log_frequency)
        wrapper.validate_model(validation_torch_dataloader)
        model_dir = MODEL_DIR.replace('[SUFFIX]', f"{marker}_{str(epoch)}").replace('[OUTPUT]', output_dir)
        wrapper.save_model(model_dir)

    labels, predictions = wrapper.validate_model(validation_torch_dataloader)
    print(classification_report([labels], [predictions]))


def get_config(language=None):
    if language == 'fr':
        return 'camembert-base'
    if language == 'en':
        return 'bert-base-cased'
    if language == 'multi':
        return 'bert-base-multilingual-cased'


if __name__ == '__main__':
    main(parse_arguments())
