#!/usr/bin/env python
from time import sleep

import pycurl
import certifi
import urllib
from io import BytesIO
import argparse

DEFAULT_OUTPUT_FILE_NAME = "halshs_complete_1mo.bib"
DEFAULT_ROWS = 10000

URL = "https://api.archives-ouvertes.fr/search/halshs/?"

FILTERS = "docType_s:(ART OR OUV OR COUV OR COMM OR THESE OR HDR OR REPORT OR NOTICE OR PROCEEDINGS)"

MAX_ATTEMPS = 10


def parse_arguments():
    parser = argparse.ArgumentParser(description='Fetches HAL SHS bibliographic references in Bibref format.')
    parser.add_argument('file', metavar='F', type=str, nargs=1,
                        help='Output file relative path')
    parser.add_argument('--start', dest='start', default=0,
                        help='Beginning offset')
    parser.add_argument('--rows', dest='rows',
                        help='Number of requested rows per request', default=DEFAULT_ROWS)
    parser.add_argument('--reset', dest='reset', action='store_true', default=False,
                        help='Whether to reset output file or no')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    start = args.start
    rows = args.rows
    reset = args.reset
    file = args.file[0]
    goon = True
    attempts = 0
    if reset:
        open(file, 'w').close()
    while goon:
        print(start)
        try:
            params = {'rows': rows, 'start': start, 'wt': 'bibtex', 'fq': FILTERS}
            request_string = URL + urllib.parse.urlencode(params)
            buffer = BytesIO()
            c = pycurl.Curl()
            c.setopt(c.URL, request_string)
            c.setopt(c.WRITEDATA, buffer)
            c.setopt(c.CAINFO, certifi.where())
            c.setopt(c.VERBOSE, True)
            c.setopt(c.CONNECTTIMEOUT, 36000)
            c.setopt(c.TIMEOUT, 36000)
            c.perform()
            c.close()
            body = buffer.getvalue()
            str_response = body.decode()
            if str_response.find("cURL error (28): Timeout was reached") >= 0:
                raise Exception("Request timeout")
            if str_response.find("cURL error") >= 0:
                raise Exception("Unknown error")
        except Exception as e:
            print(f"Request error: {e}")
            print('Wait before retry')
            sleep(10)
            attempts + 1
            if attempts >= MAX_ATTEMPS:
                print("Max number of attempts reached, abort")
                exit()
            continue
        start += rows
        goon = len(str_response) > 0
        print(str_response[1:10000])
        attempts = 0
        with open(file, "a") as output_file:
            output_file.write(str_response)
            output_file.flush()
