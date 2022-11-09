# cephal-parser

## Procédure

### 1. Récupération de HAL SHS

* [dump_hal.py](dump_hal.py)

Récupération des références de HAL SHS depuis l'API HAL vers un fichier Bibtex (.bib).

N.B. Le script est basé sur une exécution en parallèle de paquets de {rows} requêtes.

```shell
python3 dump_hal.py --help
usage: dump_hal.py [-h] [--rows ROWS] [--dir DIR]

Fetches HAL SHS bibliographic references in Bibref format.

options:
  -h, --help   show this help message and exit
  --rows ROWS  Number of requested rows per request
  --dir DIR    Output directory
```

A l'issue, vérifier et supprimer les éventuels enregistrements en erreur (l'API Hal renvoie des messages d'erreur avec
des codes HTTP 200 : ils peuvent donc être écrits dans le fichier résultat comme des réponses légitimes) :

```shell
cd my_output_directory
for file in $( fgrep -rl 'cURL error' .); do echo "$file"; done
for file in $( fgrep -rl 'cURL error' .); do rm "$file"; done
for file in $( fgrep -rl 'cURL error' .); do echo "$file"; done
```

Puis concatener tous les Bibtex :

```shell
find hal_dump -maxdepth 1 -type f -name '*.bib' -print0 | xargs -0 cat > allhal.bib
```

* [replace_in_bib.sh](replace_in_bib.sh)

Quelques remplacements dans le fichier bib (macros BibTex)

```shell
 ./replace_in_bib.sh my_dump_hal.bib 
```

* [generate_samples.py](generate_samples.py)

Génération d'un fichier auto.jsonl dans le format accepté notamment par Doccano.

* [convert_jsonl_to_iob.py](convert_jsonl_to_iob.py)

Conversion du fichier jsonl en IOB pour l'entraînement du modèle

* [bert_bibliographic_references_parsing.ipynb](bert_bibliographic_references_parsing.ipynb)

Conversion du fichier jsonl en IOB pour l'entraînement du modèle


