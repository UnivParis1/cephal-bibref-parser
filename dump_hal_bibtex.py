#!/usr/bin/env python
import argparse
import asyncio
import itertools
import os
from pathlib import Path

import httpx as httpx
import requests

DEFAULT_OUTPUT_DIR_NAME = "hal_dump"
DEFAULT_OUTPUT_FILE_NAME = "[DIR]/[DOCID].bib"
DEFAULT_ROWS = 10000

HAL_API_URL = "https://api.archives-ouvertes.fr/search/halshs/?"

LIST_QUERY_TEMPLATE = "q=docType_s:(ART OR OUV OR COUV OR COMM OR THESE OR HDR OR REPORT OR NOTICE OR PROCEEDINGS)" \
                      "&cursorMark=[CURSOR]" \
                      "&sort=docid asc&rows=[ROWS]"

BIBTEX_QUERY_TEMPLATE = "wt=bibtex&q=docid:[DOCID]"

MAX_ATTEMPTS = 10


def parse_arguments():
    parser = argparse.ArgumentParser(description='Fetches HAL SHS bibliographic references in Bibref format.')
    parser.add_argument('--rows', dest='rows',
                        help='Number of requested rows per request', default=DEFAULT_ROWS)
    parser.add_argument('--dir', dest='dir',
                        help='Output directory', default=DEFAULT_OUTPUT_DIR_NAME)
    return parser.parse_args()


async def export_file_path(doc_id, directory):
    return DEFAULT_OUTPUT_FILE_NAME.replace("[DIR]", directory).replace("[DOCID]", doc_id)


async def fetch_bibtex(doc_id, directory, client):
    file = await export_file_path(doc_id, directory)
    if os.path.exists(file):
        print(f"{doc_id} Bibtex yet present")
        return
    print(f"{doc_id} begin : export to {file}")
    bibtex_request_string = HAL_API_URL + BIBTEX_QUERY_TEMPLATE.replace('[DOCID]', str(doc_id))
    response = await client.get(bibtex_request_string, timeout=360)
    bibtex_response = response.content.decode()
    with open(file, "w") as output_file:
        output_file.write(bibtex_response)
    print(f"{doc_id} end")


async def fetch_bibtex_group(doc_ids, directory):
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(
            *map(fetch_bibtex, doc_ids, [directory] * len(doc_ids), itertools.repeat(client), )
        )


async def main(args):
    rows = args.rows
    print(f"Rows per request : {rows}")
    directory = args.dir
    Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"Output directory : {directory}")
    cursor = "*"
    total = 0
    while True:
        json_request_string = HAL_API_URL + LIST_QUERY_TEMPLATE.replace('[CURSOR]', str(cursor)).replace('[ROWS]',
                                                                                                         str(rows))
        print(f"Request to HAL : {json_request_string}")
        response = requests.get(json_request_string, timeout=360)
        json_response = response.json()
        if not total:
            total = int(json_response['response']['numFound'])
            print(f"{total} entries to fetch for this request")
        cursor = json_response['nextCursorMark']
        docs = json_response['response']['docs']
        if len(docs) == 0:
            print("Download complete !")
            break
        print(f"Group of {len(docs)} documents : begin")
        task = asyncio.create_task(fetch_bibtex_group([doc['docid'] for doc in docs], directory))
        print("Group end")
        await task


if __name__ == '__main__':
    arguments = parse_arguments()
    while True:
        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(main(arguments))
            print(result)
        except (RuntimeError, KeyError) as e:
            print(f"Error : {e}")
            continue
