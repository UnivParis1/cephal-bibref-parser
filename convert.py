import json
import re

MULTIVALUED_TAGS = ["Auteur"]


class TOKEN:
    def __init__(self, id, text, ner):
        self.id = id
        self.orth = text
        self.space = ' '
        self.tag = '-'
        self.ner = ner

    def serialize(self):
        return self.__dict__


def get_bilou_tokens(token_id, text, multi=True, label='', word_delimiter=' '):
    tokens = []
    words = [i.strip() for i in text.split(word_delimiter) if len(i) > 0]
    if multi:
        if label:
            action_tag = 'B-'
        else:
            action_tag = 'O'
    else:
        action_tag = ''

    for word in words[:-1]:
        token = TOKEN(token_id, word, action_tag + label)
        token_id += 1
        if label and multi:
            action_tag = 'I-'

        tokens.append(token)
    if multi:
        if label:
            if len(words) == 1:
                action_tag = 'U-'
            else:
                action_tag = 'L-'
    else:
        action_tag = ''

    token = TOKEN(token_id, words[-1], action_tag + label)
    tokens.append(token)
    token_id += 1

    return tokens, token_id


def get_iob_tokens(token_id, text, multi=True, label='', word_delimiter=' '):
    tokens = []
    words = [i.strip() for i in text.split(word_delimiter) if len(i) > 0]

    if multi:
        if label:
            action_tag = 'B-'
        else:
            action_tag = 'O'
    else:
        action_tag = ''

    for word in words:
        token = TOKEN(token_id, word, action_tag + label)
        token_id += 1
        if label and multi:
            action_tag = 'I-'

        tokens.append(token)

    token_id += 1

    return tokens, token_id


class SENTENCE:
    def __init__(self):
        self.tokens = []
        self.brackets = []

    def add_tokens(self, tokens):
        for token in tokens:
            self.tokens.append(token)

    def serialize(self):
        self.tokens = [i.serialize() for i in self.tokens]
        return self.__dict__


class PARAGRAPH:
    def __init__(self, raw=None):
        self.raw = raw
        self.sentences = []
        self.cats = []
        self.entities = []
        self.links = []

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def add_entity(self, entity):
        self.entities.append(entity)

    def serialize(self):
        self.sentences = [i.serialize() for i in self.sentences]
        return self.__dict__


class ANNOTATION:
    def __init__(self, id):
        self.id = id
        self.paragraphs = []

    def add_paragraph(self, paragraph):
        self.paragraphs.append(paragraph)

    def serialize(self):
        self.paragraphs = [i.serialize() for i in self.paragraphs]
        return self.__dict__


def convert_to_tags_format(jsonl_file, para_delimiter='\n\n\n', line_delimiter='\n', word_delimiter=' ', format='IOB'):
    fl = open(jsonl_file, 'r', encoding='utf8')
    lines = fl.readlines()

    annotations = []
    annotation_id = 0

    for line in lines:
        doc = json.loads(line)
        text_key = 'text' if 'text' in doc.keys() else 'data'
        text = doc[text_key]
        entity_key = 'entities' if 'entities' in doc.keys() else 'label'
        entities = doc[entity_key]
        start_key = 'start' if entity_key == 'entities' else 0
        end_key = 'end' if entity_key == 'entities' else 1
        label_key = 'label' if entity_key == 'entities' else 2

        text_without_entities = ""
        last_idx = 0

        for index, e in enumerate(entities):
            inter = text[last_idx:e[start_key]]
            # if len(inter) > 0 and last_idx > 0 and not inter.isspace():
            #     inter = ' ' + inter
            text_without_entities += inter + "_|" * (e[end_key] - e[start_key])
            # text_without_entities += inter + '_|' * (e[end_key] - e[start_key])
            last_idx = e[end_key]
            if index == len(entities) - 1:
                text_without_entities += text[last_idx:]

        token_id = 0
        entity_idx = 0

        annotation = ANNOTATION(annotation_id)
        annotation_id += 1

        paragraphs = [i for i in text_without_entities.split(para_delimiter) if len(i) > 0]
        txt_para = [i for i in text.split(para_delimiter) if len(i) > 0]

        for i, p in enumerate(paragraphs):
            paragraph = PARAGRAPH(raw=txt_para[i])
            sentences = [i for i in p.split(line_delimiter) if len(i) > 0]
            for s in sentences:
                sentence = SENTENCE()
                words = [i.strip() for i in s.split(word_delimiter) if len(i) > 0]
                for word in words:
                    if re.search('_\|{1,}', word):
                        entity = entities[entity_idx]
                        entity_idx += 1
                        converters = {
                            'BILOU': get_bilou_tokens,
                            'IOB': get_iob_tokens
                        }
                        labelled_word = text[entity[start_key]:entity[end_key]]
                        labelled_word_in_context = re.sub(r'(_\|)+', labelled_word, word)
                        tag = entity[label_key]
                        multi = tag in MULTIVALUED_TAGS
                        tokens, token_id = converters[format](token_id, labelled_word_in_context, multi,
                                                              tag, word_delimiter)
                        sentence.add_tokens(tokens)
                        paragraph.add_entity([entity[start_key], entity[end_key], tag])
                    else:
                        token = TOKEN(token_id, word, 'O')
                        sentence.add_tokens([token])
                paragraph.add_sentence(sentence)
            annotation.add_paragraph(paragraph)

        yield annotation.serialize()

