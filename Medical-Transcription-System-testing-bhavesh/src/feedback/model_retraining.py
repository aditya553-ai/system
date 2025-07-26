import os
import json
from ner.entity_recognition import fine_tune_ner_model
from normalization.terminology_resolver import fine_tune_resolver_model

def load_corrections(corrections_file):
    with open(corrections_file, 'r') as file:
        return json.load(file)

def retrain_models(corrections):
    ner_model = fine_tune_ner_model(corrections['ner'])
    resolver_model = fine_tune_resolver_model(corrections['resolver'])
    return ner_model, resolver_model

def main():
    corrections_file = os.path.join('data', 'processed', 'corrections.json')
    corrections = load_corrections(corrections_file)
    ner_model, resolver_model = retrain_models(corrections)
    print("Models retrained successfully.")

if __name__ == "__main__":
    main()