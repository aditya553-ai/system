import unittest
from src.asr.transcribe import model, whisper

class TestASR(unittest.TestCase):

    def test_load_model(self):
        self.assertIsNotNone(model, "Model should be loaded successfully.")

    def test_load_audio(self):
        audio = whisper.load_audio("Recording (7).m4a")
        self.assertIsNotNone(audio, "Audio should be loaded successfully.")

    def test_detect_language(self):
        audio = whisper.load_audio("Recording (7).m4a")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        self.assertIsNotNone(detected_lang, "Language should be detected.")

    def test_decode_audio(self):
        audio = whisper.load_audio("Recording (7).m4a")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
        self.assertIsNotNone(result.text, "Decoded text should not be None.")

if __name__ == '__main__':
    unittest.main()