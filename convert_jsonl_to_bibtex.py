import json
import re
from nameparser import HumanName
dictionnaire_etiquettes = {"Auteur": "author",
 "DatePublication": "year",
 "Titre": "title",
 "TitreRevue": "journal",
 "Volume": "volume",
 "Numero": "number",
 "Pages": "pages",
 "id": "ID",
 "type": "ENTRYTYPE",
 "DateFinColloque": "date",
 "AParaitre": "date",
 "Collection": "series",
 "EditeurScientifique": "editor",
 "TitreOuvrageCollectif": "booktitle",
 "url": "url",
 "collaborateur": "series",
 "EditeurCommerial": "publisher",
 "LieuPublication": "address",
 "DOI": "doi",
 "papges_totales": "pages",
 "Edition": "edition"
 }


#Crée un dictionnaire à partir du fichier jsonl
def jsonl_to_dict(file):
    with open(file, "r", encoding="utf-8") as fl:
        lines = fl.readlines()
        liste_publications = []
        for line in lines:
            line_dict = json.loads(line)
            opus = {}
            for label in line_dict["label"]:
                if label[-1] == "Auteur" or label == "EditeurScientifique":
                    if re.search("[A-Z]{2,}", line_dict['text'][label[0]:label[1]]) != None:
                        auteur_sans_virgule = line_dict['text'][label[0]:label[1]]
                        objet_match = re.search("([A-Z]{2,})\s[A-Z]", auteur_sans_virgule)
                        place_de_la_virgule = objet_match.span()[1] - 2
                        auteur_avec_virgule = auteur_sans_virgule[:place_de_la_virgule] +"," + auteur_sans_virgule[place_de_la_virgule:]
                        parsed_name = HumanName(auteur_avec_virgule)
                        name = parsed_name.first + " " + parsed_name.last
                    else:
                        parsed_name = HumanName(line_dict['text'][label[0]:label[1]])
                        name = parsed_name.first + " " + parsed_name.last
                    if label[-1] in opus.keys():
                        name_2 = " and " + name
                        opus[label[-1]] += name_2
                    else:
                        opus.update({label[-1]: name})
                else:
                    if label[-1] in opus.keys():
                        opus[label[-1]] = opus[label[-1]] + ", " + line_dict['text'][label[0]:label[1]]
                    else:
                        opus.update({label[-1]: line_dict['text'][label[0]:label[1]]})
            id = line_dict["id"]
            opus.update({"id":id})
            opus = {dictionnaire_etiquettes.get(k, k): v for k, v in opus.items()}
            liste_publications.append(opus)
    return liste_publications

#Calcule la typologie de document
def type_doc(dictionary):
    type_de_publi = ""
    for entry in dictionary:
        if "editor" in entry.keys():
            if "author" not in entry.keys():
                type_de_publi = "book"
            else:
                type_de_publi= "incollection"
        elif "journal" in entry.keys():
            if "editor" in entry.keys():
                if "author" in entry.keys():
                    type_de_publi = "inspecialnumber"
                else:
                    type_de_publi = "specialnumber"
            elif "editor" not in entry.keys():
                type_de_publi = "article"
        else:
            type_de_publi = "book"
        entry.update({"ENTRYTYPE":type_de_publi})
    return dictionary

def liste_string(liste_dict):
    liste_publications_chaine = []
    for dictionnaire in liste_dict:
        chaine_1 = "@" + dictionnaire["ENTRYTYPE"]
        chaine_2 = str(dictionnaire["ID"]) +","
        chaine_3 = ""
        for key, value in dictionnaire.items():
            if (key != "ENTRYTYPE") or (key != "ID"):
                chaine_3 += key + ' = {' + str(value) + "}, \n"
        chaine_4 = chaine_1 + "{" + chaine_2 + chaine_3 + "\n }"
        liste_publications_chaine.append(chaine_4)
    return(liste_publications_chaine)

liste_dictionnaires = jsonl_to_dict("C:/Users/abuccheri/Desktop/PRISM.jsonl")
liste_dictionnaires_avec_type = type_doc(liste_dictionnaires)
liste_documents_chaine = liste_string(liste_dictionnaires_avec_type)

with open('C:/Users/abuccheri/Desktop/PRISM.bib', 'w', encoding="utf-8") as f:
    for line in liste_documents_chaine:
        f.write(f"{line}\n")