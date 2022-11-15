import numpy as np
import torch
from torch.utils.data import Dataset


class BibRefParserDataset(Dataset):
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

        embedding = self.tokenizer(sentence,
                                   is_split_into_words=True,
                                   return_offsets_mapping=True,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self.max_length)

        if not labels:
            labels = [list(self.labels_to_ids.values())[-1]] * len(sentence)
        else:
            labels = [self.labels_to_ids[label] for label in labels]

        token_labels = np.ones(len(embedding["offset_mapping"]), dtype=int) * -100

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

    def __len__(self):
        return self.length
