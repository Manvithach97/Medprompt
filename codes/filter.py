def filter_entities(entities):

    imp = {"DISEASE_DISORDER","SIGN_SYMPTOM","BIOLOGICAL_STRUCTURE", "DRUG","HISTORY","TEST","VITAL_SIGN","PROCEDURE" }

    filtered =[]

    for ent in entities:
        ent_type = ent.get("entity_group")
        score = float(ent.get("score", 0))

        if ent_type in imp or score > 0.4:
            #Append a tuple of text n ent_type
            filtered.append((ent.get("word"), ent_type))  
    
    return filtered
