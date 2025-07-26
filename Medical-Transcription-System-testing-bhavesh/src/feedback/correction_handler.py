import json

class CorrectionHandler:
    def __init__(self, correction_file='corrections.json'):
        self.correction_file = correction_file
        self.load_corrections()

    def load_corrections(self):
        try:
            with open(self.correction_file, 'r') as file:
                self.corrections = json.load(file)
        except FileNotFoundError:
            self.corrections = {}

    def save_corrections(self):
        with open(self.correction_file, 'w') as file:
            json.dump(self.corrections, file, indent=4)

    def add_correction(self, entity, correction):
        self.corrections[entity] = correction
        self.save_corrections()

    def get_correction(self, entity):
        return self.corrections.get(entity, None)

    def remove_correction(self, entity):
        if entity in self.corrections:
            del self.corrections[entity]
            self.save_corrections()