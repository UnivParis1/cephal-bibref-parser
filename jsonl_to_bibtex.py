import json
def jsonl_to_dict(file):
    with open(file, "r", encoding="utf-8") as fl:
        lines = fl.readlines()
        liste_publications = []
        for line in lines:
            line_dict = json.reads(line)
            opus = {}
            for label in line_dict["label"]:
                opus.update({label[-1]: line_dict['text'][label[0]:label[1]]})
            liste_publications += opus
    return liste_publications

a = jsonl_to_dict("All_AB.jsonl")