import unittest
from src.asr.transcribe import model, whisper
from src.ner.entity_recognition import NERModel
from src.normalization.terminology_resolver import TerminologyResolver
from src.note_generation.template_filling import TemplateFiller

class TestMedicalTranscriptionPipeline(unittest.TestCase):

    def setUp(self):
        self.audio_file = "test_audio.m4a"
        self.transcribed_text = "Patient has shortness of breath and is prescribed metformin 500 mg twice daily."
        self.ner_model = NERModel()
        self.terminology_resolver = TerminologyResolver()
        self.template_filler = TemplateFiller()

    def test_asr_transcription(self):
        audio = whisper.load_audio(self.audio_file)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
        self.assertIsNotNone(result.text)

    def test_ner_recognition(self):
        entities = self.ner_model.recognize_entities(self.transcribed_text)
        self.assertIn("shortness of breath", entities['Symptoms'])
        self.assertIn("metformin 500 mg", entities['Medications'])

    def test_terminology_resolver(self):
        resolved_entities = self.terminology_resolver.resolve_entities(entities)
        self.assertIn("RxNorm", resolved_entities)

    def test_template_filling(self):
        filled_template = self.template_filler.fill_template(resolved_entities)
        self.assertIn("Subjective", filled_template)
        self.assertIn("Objective", filled_template)

if __name__ == '__main__':
    unittest.main()