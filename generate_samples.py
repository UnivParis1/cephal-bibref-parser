#!/usr/bin/env python
import random

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


if __name__ == '__main__':
    # For month names generation
    locale.setlocale(locale.LC_ALL, 'fr_FR.utf8')
    # store jsonlines
    jsonlines_container = []
    # load all citations stylesheets
    csls = glob.glob('csl/*.csl')
    random.shuffle(csls)
    # load data from all. Invalid encodings, etc. has been manually removed
    bib_source = BibTeX('halshs.bib', encoding='UTF8')
    # initialize style processor
    bib_styles = [CitationStylesStyle(csl, validate=False) for csl in csls]
    bibliographies = [CitationStylesBibliography(bib_style, bib_source,
                                                 formatter.plain) for bib_style in bib_styles]

    counter = 0

    output_file = open("auto.jsonl", "w")

    for bibliography in bibliographies:
        print("************CSL SHEET**************")
        print(bibliography.style.root.base)
        try:
            for key in bibliography.source:
                # Handle a single reference
                try:
                    if SAMPLE_MODE and random.randint(0, SAMPLE_RATE) < SAMPLE_RATE:
                        continue
                    error = False
                    # Generate litteral reference
                    citation = Citation([CitationItem(key)])
                    bibliography.register(citation)
                    text = bibliography.style.root.bibliography.layout.render_bibliography(citation.cites)
                    string_reference = str(text[0])
                    structured_reference = bibliography.source[key]
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
                                # specific processing of multivalued structured 'author' field
                                for author in field_value:
                                    position = -1
                                    names = expand_person_name(author)
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
                                            f"Author note found : <{str(author)}> in <{string_reference}> whith style <{bibliography.style.root.base}>")
                            elif field == 'issued':
                                # specific processing of monovalued structured 'issued' field (date)
                                position = -1
                                dates = expand_dates(field_value)
                                for date in dates:
                                    position = detect_substring(string_reference, date)
                                    if position >= 0:
                                        break
                                if position >= 0:
                                    labels.append(create_label(position, date, tag))
                                elif STRICT_MODE:
                                    raise MissingFieldException(
                                        f"Date note found : <{str(field_value)}> in <{string_reference}> whith style <{bibliography.style.root.base}>")
                            else:
                                # any other field than 'author' or 'issued'
                                position = -1
                                strings = expand_string(str(field_value))
                                for string in strings:
                                    position = detect_substring(string_reference, str(string))
                                    if position >= 0:
                                        break
                                if position >= 0:
                                    labels.append(create_label(position, string, tag))
                                elif STRICT_MODE:
                                    raise MissingFieldException(
                                        f"Field {field} note found : <{str(field_value)}> in <{string_reference}> whith style <{bibliography.style.root.base}>")
                        except VariableError:
                            pass
                    if len(labels) == 0:
                        # reject samples without labels
                        continue
                    labels.sort()
                    line = {"id": counter, "text": string_reference, "label": labels,
                            "csl": bibliography.style.root.base}
                    jsonlines_container.append(line)
                    print(counter)
                    counter += 1
                except MissingFieldException as mfe:
                    print(mfe)
                except DoubleFieldException as dfe:
                    print(dfe)
        except Exception as e:
            print(f">> error with {bibliography.style.root.base} : #{e}")
            continue
    # shuffle and write the result to output file
    random.shuffle(jsonlines_container)
    with jsonlines.Writer(output_file) as writer:
        for line in jsonlines_container:
            writer.write(line)
