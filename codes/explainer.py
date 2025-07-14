import json
import requests

#load local corpus
with open('local_corpus.json','r',encoding='utf-8') as f:
    raw=json.load(f) 
local_corpus = {k.lower(): v for k, v in raw.items()}

def explain_entity(entity_tuple):
    """
    entity tuple: (entity text, entity type)
    output : string explanation

    """
    text, ent_type = entity_tuple
    key=text.lower()

    #local corpus
    if key in local_corpus:
        return f"'{text}':{local_corpus[key]}"
    
    if ent_type in local_corpus:
        return f"'{text}' is a {ent_type.replace('_',' ').title()}.{local_corpus[ent_type]}"

    return None


def explain_entities(entity_list):
    return [explain_entity(ent) for ent in entity_list]