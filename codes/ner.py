from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")


nerpipe = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

#NER on input text
def recognize_entities(text):
    entities=nerpipe(text)
    structured_entities = [
        {'word': ent['word'],'entity_group': ent['entity_group'],'score': ent['score']} 
        for ent in entities
    ]
    return structured_entities

"""
Model prediction
entities=recognize_entities(text)

take an example of entities
actual_entities = [
    {"word": "arthritis", "entity_group": "DISEASE"},
    {"word": "aspirin", "entity_group": "MEDICATION"}
]

# Compare
tp = sum(1 for pred in entities 
           if any(pred["word"].lower()==actual["word"].lower() and 
                  pred["entity_group"]==actual["entity_group"]
                  for actual in actual_entities))
fp=len(entities) - tp
fn=len(actual_entities) - tp

precision = tp /(tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0


"""