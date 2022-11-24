import json

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
            if "DatePublication" in opus.keys():
                id = opus["Auteur"] + ": " + opus["DatePublication"]
            else:
                id = opus["Auteur"] + ": nd"
            opus.update({"id":id})
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
        entry.update({"type":type_de_publi})
    return dictionary

liste_dictionnaires = jsonl_to_dict("C:/Users/abuccheri/Desktop/all_AB.jsonl")
liste_dictionnaires_avec_type = type_doc(liste_dictionnaires)
