import os
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset


class BibRefParserDataset(Dataset):
    NO_LABEL_INDEX = -100

    def __init__(self, dataframe, tokenizer, labels_to_ids, ids_to_labels, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.labels_to_ids = labels_to_ids
        self.ids_to_labels = ids_to_labels
        self.max_length = max_length
        self.length = len(dataframe)

    def __getitem__(self, index):
        sentence = self.dataframe.words[index]
        labels = self.dataframe.labels[index]

        embedding = self.get_embeddings(sentence)

        if not labels:
            labels = [list(self.labels_to_ids.values())[-1]] * len(sentence)
        else:
            labels = [self.labels_to_ids[label] for label in labels]

        token_labels = np.ones(len(embedding["offset_mapping"]), dtype=int) * self.NO_LABEL_INDEX

        i = -1
        for index, offset_mapping in enumerate(embedding["offset_mapping"]):
            car = self.tokenizer.convert_ids_to_tokens(embedding.data['input_ids'][index])
            # tokenizer adds isolated spaces
            if offset_mapping[0] == 0 and offset_mapping[1] != 0 and car != '▁':
                i += 1
                if i > len(labels) - 1:
                    break
            if offset_mapping[1] != 0:
                token_labels[index] = labels[i]
            if i == -1:
                token_labels[index] = list(self.ids_to_labels.values()).index('O')

        entry = {key: torch.as_tensor(val) for key, val in embedding.items()}
        entry['labels'] = torch.as_tensor(token_labels)
        # list(zip([tokenizer.convert_ids_to_tokens(id) for id in embedding.data['input_ids']], token_labels))
        return entry

    def get_embeddings(self, sentence):
        embedding = self.tokenizer(sentence,
                                   is_split_into_words=True,
                                   return_offsets_mapping=True,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self.max_length)
        return embedding

    def __len__(self):
        return self.length


class ModelWrapper:
    DEFAULT_MAX_LENGTH = 128

    def __init__(self, model, tokenizer, device='cpu', optimizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device

    def forward(self, batch):
        input_ids = batch['input_ids'].to(self.device, dtype=torch.long)
        attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
        labels = batch['labels'].to(self.device, dtype=torch.long) if type(batch) is dict else None
        result = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return input_ids, labels, result

    def predictions_from_logits(self, logits):
        logits = logits.view(-1, self.model.config.num_labels)
        return torch.argmax(logits, axis=1)

    def compute_precision(self, labels, logits, sumed_labels=[], sumed_predictions=[]):
        targets = labels.view(-1)
        predictions = self.predictions_from_logits(logits)
        mask = targets != BibRefParserDataset.NO_LABEL_INDEX
        labels = torch.masked_select(targets, mask)
        predictions = torch.masked_select(predictions, mask)
        sumed_labels.extend(labels)
        sumed_predictions.extend(predictions)
        return accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())

    def save_model(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.tokenizer.save_vocabulary(model_dir)
        self.model.save_pretrained(model_dir)

    def train_model(self, dataloader, log_frequency=100):
        loss, precision, samples_counter, batches_counter = 0, 0, 0, 0
        self.model.train()
        batch_size = None

        for idx, batch in enumerate(dataloader):
            batch_size = batch_size or len(batch)
            input_ids = batch['input_ids'].to(self.device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
            labels = batch['labels'].to(self.device, dtype=torch.long)

            result = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = result.logits
            loss += result.loss.item()
            batches_counter += 1
            samples_counter += batch_size

            if idx % log_frequency == 0:
                loss_step = loss / batches_counter
                print(f"> {log_frequency} batches processed, loss : {loss_step}")

            precision += self.compute_precision(labels, logits)

            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(), max_norm=10
            )

            self.optimizer.zero_grad()
            result.loss.backward()
            self.optimizer.step()

        epoch_loss = loss / batches_counter
        epoch_precision = precision / batches_counter
        print(f"Epoch processed, loss : {epoch_loss}")
        print(f"Average epoch loss : {epoch_loss}")
        print(f"Average epoch precision : {epoch_precision}")

    def validate_model(self, loader):
        self.model.eval()

        batch_size = None
        loss, precision, examples_counter, batches_counter = 0, 0, 0, 0
        validation_predictions, validation_labels = [], []

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                batch_size = batch_size or len(batch)
                input_ids, labels, result = self.forward(batch)
                logits = result.logits
                loss += result.loss.item()

                batches_counter += 1
                examples_counter += batch_size

                precision += self.compute_precision(labels, logits, validation_labels, validation_predictions)

        labels = [self.model.config.id2label[label_id.item()] for label_id in validation_labels]
        predictions = [self.model.config.id2label[label_id.item()] for label_id in validation_predictions]

        loss = loss / batches_counter
        precision = precision / batches_counter
        print(f"Validation loss: {loss}")
        print(f"Validation precision : {precision}")

        return labels, predictions

    def prediction_by_tokens(self, inputs):
        input_ids, labels, result = self.forward(inputs)
        predictions = self.predictions_from_logits(result.logits)
        tokens = self.tokenizer.convert_ids_to_tokens([int(i) for i in list(input_ids[0])])
        token_predictions = [self.model.config.id2label[i] for i in predictions.numpy()]
        return tokens, token_predictions

    def predictions_by_word(self, sentence, max_length):
        inputs = self.tokenizer(sentence.split(),
                                is_split_into_words=True,
                                return_offsets_mapping=True,
                                padding='max_length',
                                truncation=True,
                                max_length=max_length,
                                return_tensors="pt")
        tokens, token_predictions = self.prediction_by_tokens(inputs)
        tokens_predictions_map = list(zip(tokens, token_predictions))
        predictions_by_word = []
        token_counter = 0
        for token_prediction, offset_mapping in zip(tokens_predictions_map,
                                                    inputs["offset_mapping"].squeeze().tolist()):
            token = tokens[token_counter]
            if offset_mapping[0] == 0 and offset_mapping[1] != 0 and token != '▁':
                predictions_by_word.append(token_prediction[1])
            token_counter += 1
        return predictions_by_word
