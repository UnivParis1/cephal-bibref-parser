import re


class TextProcessor(object):

    @classmethod
    def prepare(cls, text):
        text = ' '.join(text.split())
        text = cls.no_space_after("'#([{", text)
        text = cls.space_after("!`\"»«“”&)}]:;?/.,", text)
        text = cls.space_before("!`\"»«“”#&([{:;?/", text)
        text = cls.no_space_before("')}].,", text)
        return text.strip()

    @classmethod
    def space_after(cls, chars, text):
        return re.sub(rf"([{re.escape(chars)}])(?!\s)", '\\1 ', text)

    @classmethod
    def space_before(cls, chars, text):
        return re.sub(rf"(?<!\s)([{re.escape(chars)}])", ' \\1', text)

    @classmethod
    def no_space_after(cls, chars, text):
        return re.sub(rf"([{re.escape(chars)}])\s", '\\1', text)

    @classmethod
    def no_space_before(cls, chars, text):
        return re.sub(rf"\s([{re.escape(chars)}])", '\\1', text)


class JsonBuilder(object):

    def __init__(self, sentence: list[str], model_output: list[str]):
        self.sentence = sentence
        self.model_output = model_output
        self.output = {}
        self.inside = None
        self.buffer = []

    def empty_buffer(self):
        if self.inside is not None:
            self.output.setdefault(self.inside, [])
            self.output[self.inside].append(self.buffer)
            self.inside = None
            self.buffer = []

    def build(self) -> object:
        pairs = zip(self.sentence.split(), self.model_output)

        for pair in pairs:
            if pair[1].startswith('B-'):
                self.empty_buffer()
                key = pair[1][2:]
                self.inside = key
                self.buffer.append(pair[0])
            elif pair[1].startswith('I-'):
                key = pair[1][2:]
                if self.inside == key:
                    self.buffer.append(pair[0])
            elif pair[1] != 'O':
                self.empty_buffer()
                key = pair[1]
                self.output.setdefault(key, [])
                self.output[key].append(pair[0])
            else:
                self.empty_buffer()
        self.convert_to_dict()
        return self.output

    def convert_to_dict(self):
        for key in self.output.keys():
            self.output[key] = self.join_strings(self.output[key])

    def join_strings(self, param: list):
        if type(param[0]) == list:
            return [self.join_strings(elem) for elem in param]
        elif type(param[0]) == str:
            return ' '.join(param)
