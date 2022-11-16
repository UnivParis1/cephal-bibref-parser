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


