#!/usr/bin/env python
# coding: utf-8

import os
import sys
import itertools
import numpy as np
import pandas as pd
import datetime
from traitlets import link

from install import install

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

install('torch')
install('transformers')
install('seqeval[gpu]')
install('sklearn')

import torch
from torch import cuda
from torch.utils.data import DataLoader
from transformers import CamembertTokenizerFast, CamembertForTokenClassification
from data import BibRefParserDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive

    drive.mount('/content/drive')
    SOURCE_DATA_FILE_NAME = '/content/drive/MyDrive/bibref-parser/all_tagged_bib_refs_fr.json'
else:
    SOURCE_DATA_FILE_NAME = 'bio_tagged/all_tagged_bib_refs_fr.json'

MODEL_DIR = "/content/drive/MyDrive/bibref-parser/output/[SUFFIX]"

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

e = datetime.datetime.now()
marker = "%s-%s-%s-%s-%s-%s" % (e.year, e.month, e.day, e.hour, e.minute, e.second)

data = pd.read_json(SOURCE_DATA_FILE_NAME)
# data = data[0:50000]
print(data.head())
print(data.shape)

unique_labels = np.unique(list(itertools.chain(*data.labels)))
labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
ids_to_labels = {v: k for v, k in enumerate(unique_labels)}
unique_labels

MAX_LENGTH = 128
TRAINING_EPOCHS = 10
TRAINING_BATCH_SIZE = 4
VALIDATION_BATCH_SIZE = 2
LEARNING_RATE = 5e-06
TEST_SIZE = 0.1

tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

training_dataset, testing_dataset = train_test_split(data, test_size=TEST_SIZE)
training_dataset = training_dataset.reset_index(drop=True)
testing_dataset = testing_dataset.reset_index(drop=True)
training_torch_dataset = BibRefParserDataset(training_dataset, tokenizer, labels_to_ids, ids_to_labels, MAX_LENGTH)
validation_torch_dataset = BibRefParserDataset(testing_dataset, tokenizer, labels_to_ids, ids_to_labels, MAX_LENGTH)

training_torch_dataloader = DataLoader(training_torch_dataset, **{'batch_size': TRAINING_BATCH_SIZE,
                                                                  'shuffle': True,
                                                                  'num_workers': 0
                                                                  })
validation_torch_dataloader = DataLoader(validation_torch_dataset, **{'batch_size': VALIDATION_BATCH_SIZE,
                                                                      'shuffle': True,
                                                                      'num_workers': 0
                                                                      })
bibref_parser_model = CamembertForTokenClassification.from_pretrained('camembert/camembert-large',
                                                                      num_labels=len(labels_to_ids))
bibref_parser_model.to(device)
optimizer = torch.optim.Adam(params=bibref_parser_model.parameters(), lr=LEARNING_RATE)


def forward(batch, model):
    input_ids = batch['input_ids'].to(device, dtype=torch.long)
    attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
    labels = batch['labels'].to(device, dtype=torch.long) if type(batch) is dict else None
    result = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return input_ids, labels, result


def predictions_from_logits(logits):
    logits = logits.view(-1, bibref_parser_model.num_labels)
    predictions = torch.argmax(logits, axis=1)
    return predictions


def compute_precision(labels, logits, sumed_labels=[], sumed_predictions=[]):
    targets = labels.view(-1)
    predictions = predictions_from_logits(logits)
    precision = labels.view(-1) != -100
    labels = torch.masked_select(targets, precision)
    predictions = torch.masked_select(predictions, precision)
    sumed_labels.extend(labels)
    sumed_predictions.extend(predictions)
    return accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())


def train():
    loss, precision, examples_counter, steps_counter = 0, 0, 0, 0
    bibref_parser_model.train()

    for idx, batch in enumerate(training_torch_dataloader):
        unput_ids = batch['input_ids'].to(device, dtype=torch.long)
        attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
        labels = batch['labels'].to(device, dtype=torch.long)

        result = bibref_parser_model(input_ids=unput_ids, attention_mask=attention_mask, labels=labels)
        logits = result.logits
        loss += result.loss.item()
        steps_counter += 1
        examples_counter += TRAINING_BATCH_SIZE

        if idx % 100 == 0:
            loss_step = loss / steps_counter
            print(f"Training loss per 100 training steps: {loss_step}")

        precision += compute_precision(labels, logits)

        torch.nn.utils.clip_grad_norm_(
            parameters=bibref_parser_model.parameters(), max_norm=10
        )

        optimizer.zero_grad()
        result.loss.backward()
        optimizer.step()

    epoch_loss = loss / steps_counter
    precision = precision / steps_counter
    print(f"Loss: {epoch_loss}")
    print(f"Precision: {precision}")


def save_model(suffix, tokenizer, model):
    model_dir = MODEL_DIR.replace('[SUFFIX]', suffix)
    os.makedirs(model_dir, exist_ok=True)
    tokenizer.save_vocabulary(model_dir)
    model.save_pretrained(model_dir)


# In[ ]:


def validate(model, loader):
    model.eval()

    loss, precision, examples_counter, steps_counter = 0, 0, 0, 0
    validation_predictions, validation_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            input_ids, labels, result = forward(batch, bibref_parser_model)
            logits = result.logits
            loss += result.loss.item()

            steps_counter += 1
            examples_counter += VALIDATION_BATCH_SIZE

            precision += compute_precision(labels, logits, validation_labels, validation_predictions)

    labels = [ids_to_labels[label_id.item()] for label_id in validation_labels]
    predictions = [ids_to_labels[label_id.item()] for label_id in validation_predictions]

    loss = loss / steps_counter
    precision = precision / steps_counter
    print(f"Loss: {loss}")
    print(f"Accuracy: {precision}")

    return labels, predictions


# In[ ]:
torch.cuda.empty_cache()
for epoch in range(TRAINING_EPOCHS):
    print(f"Beginning epoch n°: {epoch + 1}")
    train()
    validate(bibref_parser_model, validation_torch_dataloader)
    save_model(f"{marker}_{str(epoch)}", tokenizer, bibref_parser_model)

# In[ ]:


labels, predictions = validate(bibref_parser_model, validation_torch_dataloader)

# In[ ]:


from seqeval.metrics import classification_report

print(classification_report([labels], [predictions]))

# In[ ]:


sentence = "Bernard Gauriau. La consécration jurisprudentielle de la représentation syndicale de groupe et de l'accord de groupe. Droit Social, Dalloz, 2003, 07 et 08, pp.732."
sentence = "LE MEUR, Pierre-Yves, Anthropologie de la gouvernance. Politique des ressources, dispositifs du développement et logiques d'acteurs,, 2006."
sentence = "LEDOUX Sébastien, « La mémoire, mauvais objet de l’historien ? », Vingtième siècle. Revue d’histoire, n°133, janvier-mars 2017, p.113-"
sentence = "BOULLAND Paul (dir.), Dictionnaire biographique des militants des industries électriques et gazières. De la Libération aux années 2000, Ivry-sur-Seine, Éditions de l’Atelier, décembre 2013, 462 p."
sentence = "VERLAINE Julie, « La destinée singulière d’un peintre face à l’évolution du goût artistique (1945-1969) », dans Dominique Gagneux (dir.), Serge Poliakoff. Le rêve des formes, catalogue d’exposition, Paris, Musée d’art moderne de la Ville de Paris \/ Paris-Musées, 2013, p. 77-"
bibref_parser_model.eval()

inputs = tokenizer(sentence.split(),
                   is_split_into_words=True,
                   return_offsets_mapping=True,
                   padding='max_length',
                   truncation=True,
                   max_length=MAX_LENGTH,
                   return_tensors="pt")

input_ids, labels, result = forward(inputs, bibref_parser_model)

predictions = predictions_from_logits(result.logits)

tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
token_predictions = [ids_to_labels[i] for i in predictions.cpu().numpy()]
tokens_predictions_map = list(zip(tokens, token_predictions))
print(tokens_predictions_map)

predictions_by_word = []
token_counter = 0
for token_prediction, offset_mapping in zip(tokens_predictions_map, inputs["offset_mapping"].squeeze().tolist()):
    token = tokens[token_counter]
    if offset_mapping[0] == 0 and offset_mapping[1] != 0 and token != '▁':
        predictions_by_word.append(token_prediction[1])
    token_counter += 1

print(list(zip(sentence.split(), predictions_by_word)))

# In[ ]:


len(unique_labels)
labels_to_ids
