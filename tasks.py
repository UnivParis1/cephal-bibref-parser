from celery import Celery
from celery.signals import worker_process_init
from dotenv import dotenv_values
from transformers import CamembertTokenizerFast, CamembertForTokenClassification

from data import ModelWrapper
from utils import TextProcessor, JsonBuilder

PARSER_MODEL_PATH = './model'

celery_params = dict(dotenv_values(".env.celery"))

app = Celery('tasks', **celery_params)

def initialization():
    proxies = dict(dotenv_values(".env.proxies"))

    tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base", proxies=proxies)

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

    output = JsonBuilder(reference, predictions_by_word).build()

    return output
