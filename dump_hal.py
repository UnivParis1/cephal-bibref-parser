#!/usr/bin/env python
from time import sleep

import requests
import argparse
from timeit import default_timer as timer

DEFAULT_OUTPUT_FILE_NAME = "halshs_complete.bib"
DEFAULT_ROWS = 10000

HAL_API_URL = "https://api.archives-ouvertes.fr/search/halshs/?"

LIST_QUERY_TEMPLATE = "q=docType_s:(ART OR OUV OR COUV OR COMM OR THESE OR HDR OR REPORT OR NOTICE OR PROCEEDINGS)" \
                      f"&cursorMark=[CURSOR]&sort=docid asc&rows=1000"
BIBTEX_QUERY_TEMPLATE = "wt=bibtex&q=docid:[DOCID]"

MAX_ATTEMPS = 10


def parse_arguments():
    parser = argparse.ArgumentParser(description='Fetches HAL SHS bibliographic references in Bibref format.')
    parser.add_argument('file', metavar='F', type=str, nargs=1,
                        help='Output file relative path')
    parser.add_argument('--rows', dest='rows',
                        help='Number of requested rows per request', default=DEFAULT_ROWS)
    parser.add_argument('--reset', dest='reset', action='store_true', default=False,
                        help='Whether to reset output file or no')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    rows = args.rows
    reset = args.reset
    file = args.file[0]
    cursor = "*"
    attempts = 0
    total = 0
    processed = 0
    if reset:
        open(file, 'w').close()
    prev_top = top = start = timer()
    while True:
        list_request_string = HAL_API_URL + LIST_QUERY_TEMPLATE.replace('[CURSOR]', str(cursor)).replace('[ROWS]',
                                                                                                         str(rows))
        response = requests.get(list_request_string, timeout=360)
        json_response = response.json()
        if not total:
            total = int(json_response['response']['numFound'])
        cursor = json_response['nextCursorMark']
        docs = json_response['response']['docs']
        if len(docs) == 0:
            print("Dowload complete !")
            break
        for doc in docs:
            docid = doc['docid']
            list_request_string = HAL_API_URL + BIBTEX_QUERY_TEMPLATE.replace('[DOCID]', str(docid))
            response = requests.get(list_request_string, timeout=360)
            bibtex_response = response.content.decode()
            with open(file, "a") as output_file:
                output_file.write(bibtex_response)
                processed += 1
                prev_top = top
                top = timer()
                elapsed = top - start
                average = elapsed / processed
                duration = top - prev_top
                prevision = average * total
                print(f"{processed}/{total} : {docid}")
                if processed % 100 == 0:
                    print(f"(time: {duration}, moy.: {average}, total: {elapsed}, prevision: {prevision}")
