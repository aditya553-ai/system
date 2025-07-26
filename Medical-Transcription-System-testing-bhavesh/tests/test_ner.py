import unittest
from src.ner.entity_recognition import EntityRecognizer
from src.ner.error_mitigation import ErrorMitigator

class TestEntityRecognition(unittest.TestCase):

    def setUp(self):
        self.recognizer = EntityRecognizer()
        self.mitigator = ErrorMitigator()

    def test_entity_recognition(self):
        text = "The patient has type II diabetes mellitus and is prescribed metformin 500 mg."
        entities = self.recognizer.recognize_entities(text)
        expected_entities = {
            "diagnoses": ["type II diabetes mellitus"],
            "medications": ["metformin 500 mg"]
        }
        self.assertEqual(entities, expected_entities)

    def test_error_mitigation(self):
        incorrect_text = "The patint has type II dibetes mellitus."
        corrected_text = self.mitigator.mitigate_errors(incorrect_text)
        self.assertNotIn("patint", corrected_text)
        self.assertNotIn("dibetes", corrected_text)

if __name__ == '__main__':
    unittest.main()