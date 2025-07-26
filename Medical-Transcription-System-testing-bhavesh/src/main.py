import os
from asr.transcribe import transcribe_audio
from ner.entity_recognition import recognize_entities
from normalization.terminology_resolver import resolve_terms
from note_generation.template_filling import fill_template
from feedback.correction_handler import handle_corrections
from utils.metrics import calculate_metrics

def main():
    # Step 1: Audio Capture & Preprocessing
    audio_file = "data/raw/recording.m4a"
    transcribed_text = transcribe_audio(audio_file)

    # Step 2: Medical Named Entity Recognition
    entities = recognize_entities(transcribed_text)

    # Step 3: Entity Normalization & Mapping
    normalized_entities = resolve_terms(entities)

    # Step 4: Template Filling & Note Generation
    note_template = "config/templates.json"
    filled_note = fill_template(normalized_entities, note_template)

    # Step 5: Feedback Loop & Continuous Improvement
    handle_corrections(filled_note)

    # Calculate and log metrics
    metrics = calculate_metrics(transcribed_text, entities, normalized_entities)
    print(metrics)

if __name__ == "__main__":
    main()