from celery import Celery
from celery.signals import worker_process_init
from transformers import CamembertTokenizerFast, CamembertForTokenClassification

from data import ModelWrapper
from utils import TextProcessor

PARSER_MODEL_PATH = './model'

app = Celery('tasks',
             broker='redis://localhost:6379/1',
             backend='rpc://')


def initialization():
    tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

    bibref_parser_model = CamembertForTokenClassification.from_pretrained(PARSER_MODEL_PATH)

    initialization.parser_model_wrapper = ModelWrapper(bibref_parser_model, tokenizer)


@worker_process_init.connect()
def setup(**kwargs):
    print('initializing bibliographic references parser model')
    initialization()
    print('done initializing bibliographic references parser model')


@app.task
def predict_fields(reference, max_length=ModelWrapper.DEFAULT_MAX_LENGTH):
    reference = TextProcessor.prepare(reference)

    predictions_by_word = initialization.parser_model_wrapper.predictions_by_word(reference, max_length)

    return predictions_by_word
