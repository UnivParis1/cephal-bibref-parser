# cephal-parser

## Procédure

* [dump_hal.py](dump_hal.py)

Récupération des références de HAL SHS depuis l'API HAL vers un fichier Bibtex (.bib).

N.B. Le script est basé sur une exécution en parallèle de paquets de {rows} requêtes.

```
python3 dump_hal.py --help
usage: dump_hal.py [-h] [--rows ROWS] [--dir DIR]

Fetches HAL SHS bibliographic references in Bibref format.

options:
  -h, --help   show this help message and exit
  --rows ROWS  Number of requested rows per request
  --dir DIR    Output directory
```

* [replace_in_bib.sh](replace_in_bib.sh)

Quelques remplacements dans le fichier bib (macros BibTex)

* [generate_samples.py](generate_samples.py)

Génération d'un fichier auto.jsonl dans le format accepté notamment par Doccano.


* [convert_jsonl_to_iob.py](convert_jsonl_to_iob.py)

Conversion du fichier jsonl en IOB pour l'entraînement du modèle

* [bert_bibliographic_references_parsing.ipynb](bert_bibliographic_references_parsing.ipynb)

Conversion du fichier jsonl en IOB pour l'entraînement du modèle


