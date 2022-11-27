import json
import bibtexparser

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
            opus = {"Auteur" : ""}
            for label in line_dict["label"]:
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
        if "EditeurScientifique" in entry.keys():
            if len(entry["Auteur"]) == 0:
                type_de_publi = "book"
            else:
                type_de_publi= "incollection"
        elif "TitreRevue" in entry.keys():
            if "EditeurScientifique" in entry.keys():
                if len(entry["Auteur"] == 0 ):
                    type_de_publi = "specialnumber"
                else:
                    type_de_publi = "inspecialnumber"
            elif "EditeurScientifique" not in entry.keys():
                type_de_publi = "article"
        else:
            type_de_publi = "book"
        entry.update({"ENTRYTYPE":type_de_publi})
    return dictionary

def liste_string(liste_dict):
    liste_publications_chaine = []
    for line in liste_dict:
        chaine = ""
        chaine += "@" + line["ENTRYTYPE"] + "{" + str(line["ID"])
        chaine_2 = ""
        for key, value in line.items():
            chaine_2 += str(key) + ' = {' + str(value) + "}, "
        chaine_3 = chaine + chaine_2
        liste_publications_chaine.append(chaine_3)
    return(liste_publications_chaine)

liste_dictionnaires = jsonl_to_dict("C:/Users/abuccheri/Desktop/all_AB.jsonl")
liste_dictionnaires_avec_type = type_doc(liste_dictionnaires)
liste_documents_chaine = liste_string(liste_dictionnaires_avec_type)