# cephal-parser

## Procédure

### 1. Récupération de HAL SHS

* [dump_hal.py](dump_hal.py)

Récupération des références de HAL SHS depuis l'API HAL vers un fichier Bibtex (.bib).

N.B. Le script est basé sur une exécution en parallèle de paquets de {rows} requêtes.

```shell
python3 dump_hal_bibtex.py --help
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

### 2. Préparation des données pour l'entraînement du modèle

* [replace_in_bib.sh](replace_in_bib.sh)

Quelques remplacements dans le fichier bib (macros BibTex)

```shell
 ./replace_in_bib.sh my_dump_hal.bib 
```

* [convert_bibtex_to_jsonl.py](convert_bibtex_to_jsonl.py)

Génération d'un fichier auto.jsonl dans le format accepté notamment par Doccano.

```shell
$ python3 convert_bibtex_to_jsonl.py -h
usage: convert_bibtex_to_jsonl.py [-h] [--csl CSL] [--source SOURCE] [--dest DEST] [--sample SAMPLE_MODE]

Fetches HAL SHS bibliographic references in Bibref format.

options:
  -h, --help            show this help message and exit
  --csl CSL             CSL files directory path
  --source SOURCE       Bibtex source file
  --dest DEST           Jsonl destination file
  --sample SAMPLE_MODE  Process one of 100 references only
```

* [convert_jsonl_to_iob.py](convert_jsonl_to_iob.py)

Conversion du fichier jsonl en IOB pour l'entraînement du modèle.

Le script génère plusieurs fichiers (un par langue rencontrée dans les titres). Le format obtenu s'éloigne d'IOB en ceci
que seuls les champs considérés comme multivalués (voir la constante MULTIVALUED_TAGS) sont préfixés par B- ou I-.

```shell
$ python3 convert_jsonl_to_iob.py --help

usage: convert_jsonl_to_iob.py [-h] [--input INPUT] [--output-dir OUTPUT_DIR] [--output-filename OUTPUT_FILENAME_PREFIX]

Convert jsonl file to IOB format.

options:
  -h, --help            show this help message and exit
  --input INPUT         Input sentence
  --output-dir OUTPUT_DIR
                        Output directory
  --output-filename OUTPUT_FILENAME_PREFIX
                        Output file name prefix
```

### 3. Entraînement du modèle

* [bert_bibref_parsing_parser_training.py](bert_bibref_parsing_parser_training.py)

Code d'entraînement du modèle

```shell
$ python3 bert_bibref_parsing_parser_training.py --help

usage: bert_bibref_parsing_parser_training.py [-h] [--lng {fr,en,multi}] [--output OUTPUT] [--source SOURCE] [--epochs EPOCHS] [--max-length MAX_LENGTH] [--learning-rate LEARNING_RATE]
                                              [--training-batch-size TRAINING_BATCH_SIZE] [--validation-batch-size VALIDATION_BATCH_SIZE] [--validation-size VALIDATION_SIZE] [--log-frequency LOG_FREQUENCY]
                                              [--limit LIMIT]

Fine tune Bert or Camembert for bibliographic references parsing.

optional arguments:
  -h, --help            show this help message and exit
  --lng {fr,en,multi}   Expected language for titles
  --output OUTPUT       Output directory path
  --source SOURCE       IOB source file
  --epochs EPOCHS       Number of training epochs
  --max-length MAX_LENGTH
                        Wordpiece tokenized sentence max length
  --learning-rate LEARNING_RATE
                        Learning rate
  --training-batch-size TRAINING_BATCH_SIZE
                        Training batch size
  --validation-batch-size VALIDATION_BATCH_SIZE
                        Validation batch size
  --validation-size VALIDATION_SIZE
                        Validation dataset size
  --log-frequency LOG_FREQUENCY
                        Frequency of logs during training (number of batches)
  --limit LIMIT         Number of samples to take from source file

```

### 4. Prédiction

#### 4.a Test de la prédition (ligne de commande)

* [bert_bibref_parser_predict.py](bert_bibref_parser_predict.py)

Code pour la prédiction.

Il est conseillé de laisser le paramètre max length par défaut (128) ou de le régler sur la valeur qui a servi à
l'entraînement du modèle.

```shell
$ python3 bert_bibref_parser_predict.py --help

usage: bert_bibref_parser_predict.py [-h] --input INPUT [--max-length MAX_LENGTH]

Fine tune Camembert for bibliographic references parsing.

options:
  -h, --help            show this help message and exit
  --input INPUT         Input sentence
  --max-length MAX_LENGTH
                        Wordpiece tokenized sentence max length

```

#### 4.b Lancement de la prédiction (service)

* Lancement du parser sous Celery (à daemonizer de préférence)

```shell
celery -A tasks worker -l info
```

* Appel du service

```shell
$ python3 parse_reference.py --help
usage: parse_reference.py [-h] --reference REFERENCE

Call Laelaps bibliographic reference parsing task.

options:
  -h, --help            show this help message and exit
  --reference REFERENCE
                        Full text bibligraphic reference

```
