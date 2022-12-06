import argparse
import json
from tasks import predict_fields
from utils import JsonBuilder


def parse_arguments():
    parser = argparse.ArgumentParser(description='Call Laelaps bibliographic reference parsing task.')
    parser.add_argument('--reference', dest='reference', required=True,
                        help='Full text bibligraphic reference')
    args, unknown = parser.parse_known_args()
    return args


def main(arguments):
    predictions_by_word = predict_fields.delay(reference=arguments.reference).get(timeout=3)
    output = JsonBuilder(arguments.reference, predictions_by_word).build()
    print(json.dumps(output))


if __name__ == '__main__':
    main(parse_arguments())
