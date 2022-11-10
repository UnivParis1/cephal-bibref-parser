#!/usr/bin/env python
import argparse
import random

import citeproc.source
import jsonlines

import glob

import re

import calendar
import locale

from citeproc.source.bibtex import BibTeX
from citeproc.source import VariableError

from citeproc import CitationStylesStyle, CitationStylesBibliography
from citeproc import formatter
from citeproc import Citation, CitationItem

from utils import TextProcessor

DEFAULT_BIB_SOURCE_FILE = 'source.bib'
DEFAULT_JSONL_DEST_FILE = 'dest.jsonl'
DEFAULT_CSL_DIR = 'csl'

MAX_ATTEMPTS = 10

STRICT_MODE = True
SAMPLE_MODE = False
SAMPLE_RATE = 100

LABELS = {'AParaitre': None,
          'Auteur': 'author',
          'DatePublication': 'issued',
          'EditeurCommercial': 'publisher',
          'EditeurScientifique': 'editor',
          'LieuPublication': 'publisher_place',
          'Numero': 'issue',
          'Pages': 'page',
          'Titre': 'title',
          'TitreOuvrageCollectif': 'container_title',
          'TitreRevue': 'container_title',
          'Volume': 'volume'}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Fetches HAL SHS bibliographic references in Bibref format.')
    parser.add_argument('--csl', dest='csl',
                        help='CSL files directory path', default=DEFAULT_CSL_DIR)
    parser.add_argument('--source', dest='source',
                        help='Bibtex source file', default=DEFAULT_BIB_SOURCE_FILE)
    parser.add_argument('--dest', dest='dest',
                        help='Jsonl destination file', default=DEFAULT_JSONL_DEST_FILE)
    parser.add_argument('--sample', dest='sample_mode',
                        help='Process one of 100 references only', default=False)
    return parser.parse_args()


def create_label(position, value, tag):
    return [position, position + len(str(value)), tag]


def initials(gname):
    if len(gname) == 0:
        return
    initial = ''.join([name[0].upper() + "." for name in re.split('-| ', gname)])
    return initial


def dash_initials(gname):
    if len(gname) == 0:
        return
    initial = '-'.join([name[0].upper() + "." for name in re.split('-| ', gname)])
    return initial


def expand_person_name(author):
    given_name = author['given']
    if 'non-dropping-particle' in author:
        particle = author['non-dropping-particle']
        family_name = f"{particle} {author['family']}"
    else:
        family_name = author['family']
    names = name_variants(family_name, given_name)
    family_name = family_name.upper()
    names += name_variants(family_name, given_name)
    return names


def name_variants(family_name, given_name):
    variants = [str(f"{family_name} {given_name}"),
                str(f"{family_name}, {given_name}"),
                str(f"{family_name} ({given_name})"),
                str(f"{family_name} {given_name[0]}"),
                str(f"{family_name} {given_name[0]}."),
                str(f"{family_name}, {given_name[0]}"),
                str(f"{family_name}, {given_name[0]}."),
                str(f"{family_name} {initials(given_name)}"),
                str(f"{family_name} {dash_initials(given_name)}"),
                str(f"{family_name} ({initials(given_name)})"),
                str(f"{family_name} ({dash_initials(given_name)})"),
                str(f"{given_name[0]}. {family_name}"),
                str(f"{initials(given_name)} {family_name}"),
                str(f"{dash_initials(given_name)} {family_name}"),
                str(f"{given_name} {family_name}"),
                ]
    variants.sort(key=len, reverse=True)
    return variants


def expand_dates(structured_date):
    """Generates several date formats that you may encounter in bibliographical references"""
    year = structured_date['year']
    month = structured_date['month']
    dates = [f"{month} {year}", str(year)]
    if type(int(month)) == int and int(month) > 0 and int(month) < 13:
        month_name = calendar.month_name[int(month)]
        dates = [f"{month_name.capitalize()} {year}", f"{month_name} {year}"] + dates
    return dates


def expand_string(string):
    """Generates variant for strings : with not breakable dashes"""
    return [string,
            string.replace('-', u"\u2011")]


def detect_substring(global_string, substring):
    if global_string.count(substring) > 1:
        raise DoubleFieldException(f"{substring} detected twice or more in {global_string}")
    return global_string.find(substring)


class Error(Exception):
    """Base class for other exceptions"""
    pass


class MissingFieldException(Error):
    """Raised when a field is missing from generated reference"""
    pass


class DoubleFieldException(Error):
    """Raised when a field is present twice in generated reference"""
    pass


def generate_bibliographies(csl_dir: str, source_file: str) -> list[CitationStylesBibliography]:
    """
    Generate citeproc objetcs from files

    :rtype: list[CitationStylesBibliography]
    :param csl_dir: CSL stylesheets directory
    :param source_file: Bibtex source file
    :return:
    """
    # load all citations stylesheets
    csl_stylesheets = glob.glob(f"{csl_dir}/*.csl")
    random.shuffle(csl_stylesheets)
    # load data from all. Invalid encodings, etc. has been manually removed
    bib_source = BibTeX(source_file, encoding='UTF8')
    # initialize style processor
    bib_styles = [CitationStylesStyle(csl, validate=False) for csl in csl_stylesheets]
    bibliographies = [CitationStylesBibliography(bib_style, bib_source,
                                                 formatter.plain) for bib_style in bib_styles]
    return bibliographies


def json_line(bibliographies: list[CitationStylesBibliography], key: object,
              wrong_csl: list, sample_mode: bool, id: int) -> object:
    line = None
    attempts = 0
    # try to generate citation with random stylesheet and abort if MAX_ATTEMPTS is reached
    while line is None and attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            if sample_mode and random.randint(0, SAMPLE_RATE) < SAMPLE_RATE:
                continue
            print(f"Attenpt n°{attempts} CSL sheet")
            chosen_bibliography, string_reference, structured_reference = run_random_csl(bibliographies, key)
            labels = generate_labels(chosen_bibliography, string_reference, structured_reference)
            if len(labels) == 0:
                # reject samples without labels
                continue
            # check that label are not overlapping or neighbours
            overlaps = [labels[i] for i in range(1, len(labels) - 1) if int(labels[i][0]) <= int(labels[i - 1][1])]
            assert len(overlaps) == 0
            line = {"id": id, "text": string_reference, "label": labels,
                    "csl": chosen_bibliography.style.root.base}
        except MissingFieldException as mfe:
            print(mfe)
        except DoubleFieldException as dfe:
            print(dfe)
        except Exception as e:
            print(f">> error with {chosen_bibliography.style.root.base} : #{e}")
            if chosen_bibliography.style.root.base not in wrong_csl:
                wrong_csl.append(chosen_bibliography.style.root.base)
        finally:
            continue
    return line


def generate_labels(chosen_bibliography: CitationStylesBibliography, string_reference: str,
                    structured_reference: citeproc.source.Reference) -> list:
    print(f"Label generation whith style <{chosen_bibliography.style.root.base}>")
    labels = []
    for tag, field in LABELS.items():
        if field is None:
            continue
        if field not in structured_reference:
            continue
        if tag == 'TitreRevue' and not structured_reference['type'] == 'article-journal':
            continue
        if tag == 'TitreOuvrageCollectif' and not structured_reference['type'] == 'chapter':
            continue
        try:
            field_value = getattr(structured_reference, field)
            if field_value in (None, ''):
                continue
            if field == 'author':
                new_labels = generate_author_labels(field_value, string_reference, tag)
            elif field == 'issued':
                new_labels = generate_issued_label(field_value, string_reference, tag)
            else:
                new_labels = generate_default_labels(field, field_value, string_reference, tag)
            labels += new_labels
        except VariableError:
            pass
    return sorted(labels)


def generate_default_labels(field: str, field_value: list[dict], string_reference: str, tag: str) -> list[list]:
    """
            Processing of any other field than 'author' or 'issued'

            :param field_value: value of the Bibtext field
            :param string_reference: full text reference
            :param tag: tag name
            :return:
            """
    labels = []
    position = -1
    strings = [TextProcessor.prepare(string) for string in expand_string(str(field_value))]
    for string in strings:
        position = detect_substring(string_reference, str(string))
        if position >= 0:
            break
    if position >= 0:
        labels.append(create_label(position, string, tag))
    elif STRICT_MODE and False:
        raise MissingFieldException(
            f"Field {field} note found : <{str(field_value)}> in <{string_reference}>")
    return labels


def generate_issued_label(field_value: list[dict], string_reference: str, tag: str) -> list[list]:
    """
        Specific processing of monovalued structured 'issued' field (date)

        :param field_value: value of the Bibtext field
        :param string_reference: full text reference
        :param tag: tag name
        :return:
        """

    labels = []
    position = -1
    dates = [TextProcessor.prepare(date) for date in expand_dates(field_value)]
    for date in dates:
        position = detect_substring(string_reference, date)
        if position >= 0:
            break
    if position >= 0:
        labels.append(create_label(position, date, tag))
    elif STRICT_MODE:
        raise MissingFieldException(
            f"Date note found : <{str(field_value)}> in <{string_reference}>")
    return labels


def generate_author_labels(field_value: list[dict], string_reference: str, tag: str) -> list[list]:
    """
    Specific processing of multivalued structured 'author' field

    :param field_value: value of the Bibtext field
    :param string_reference: full text reference
    :param tag: tag name
    :return:
    """
    labels = []
    for author in field_value:
        position = -1
        names = [TextProcessor.prepare(name) for name in expand_person_name(author)]
        for name in names:
            position = detect_substring(string_reference, name)
            if position == -1:
                position = detect_substring(string_reference, name.upper())
            if position >= 0:
                break
        if position >= 0:
            labels.append(create_label(position, name, tag))
        elif STRICT_MODE:
            raise MissingFieldException(
                f"Author note found : <{str(author)}> in <{string_reference}>")
    return labels


def run_random_csl(bibliographies: list[CitationStylesBibliography], key: str) -> tuple[
    citeproc.frontend.CitationStylesBibliography, str, citeproc.source.Reference]:
    """
    Run random CSL stylesheet on reference

    :param bibliographies: List af avalaible bibliographies
    :param key: key of the reference to process
    :return: bibliography used, full text reference, structured reference
    """
    # Generate literal reference
    citation = Citation([CitationItem(key)])
    chosen_bibliography = bibliographies[random.randint(0, len(bibliographies) - 1)]
    print(chosen_bibliography.style.root.base)
    chosen_bibliography.register(citation)
    text = chosen_bibliography.style.root.bibliography.layout.render_bibliography(citation.cites)
    string_reference = TextProcessor.prepare(str(text[0]))
    structured_reference = chosen_bibliography.source[key]
    return chosen_bibliography, string_reference, structured_reference


def main(arguments: argparse.Namespace) -> None:
    csl_dir = arguments.csl
    source_file = arguments.source
    dest_file = arguments.dest
    sample_mode = arguments.sample_mode

    # list of failed csl
    wrong_csl = []

    # For month names generation
    locale.setlocale(locale.LC_ALL, 'fr_FR.utf8')

    bibliographies = generate_bibliographies(csl_dir, source_file)

    # use any bibliography as index
    index_bibliography = bibliographies[0]

    # store jsonlines
    jsonlines_container = [json_line(bibliographies, key, wrong_csl, sample_mode, index) for index, key in
                           enumerate(index_bibliography.source)]
    jsonlines_container = list(filter(lambda item: item is not None, jsonlines_container))
    # shuffle and write the result to output file
    random.shuffle(jsonlines_container)
    output_file = open(dest_file, "w")
    with jsonlines.Writer(output_file) as writer:
        for line in jsonlines_container:
            writer.write(line)
    print(sorted(wrong_csl))


if __name__ == '__main__':
    main(parse_arguments())
