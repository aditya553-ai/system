# Medical Transcription Pipeline

This project implements a comprehensive medical transcription pipeline designed to facilitate the transcription of clinician-patient interactions, extract relevant medical entities, and generate structured clinical notes. The pipeline is structured into several key stages:

## Stages of the Pipeline

1. **Generic ASR Transcription**
   - Captures audio of clinician-patient interactions.
   - Preprocesses audio for noise reduction and silence trimming.
   - Utilizes an ASR engine to convert audio to raw text.
   - Optionally applies punctuation restoration to enhance text quality.

2. **Medical Named Entity Recognition (NER)**
   - Fine-tunes a clinical NER model to identify and tag medical entities in the transcript.
   - Recognizes various entity categories such as symptoms, diagnoses, medications, dosages, and lab tests.
   - Implements error mitigation techniques to correct ASR mis-transcriptions.

3. **Entity Normalization & Mapping**
   - Employs terminology resolver models to map raw entity text to standardized codes.
   - Utilizes multi-terminology resolvers to align entities with various medical coding systems.
   - Assigns confidence scores to mappings, flagging low-confidence matches for review.

4. **Template Filling & Note Generation**
   - Defines structured templates for clinical notes.
   - Populates templates with normalized entities and associated metadata.
   - Integrates safety alerts to flag potential contraindications or allergies.
   - Renders a human-readable draft for clinician review.

5. **Feedback Loop & Continuous Improvement**
   - Captures clinician corrections to improve model accuracy.
   - Periodically retrains models based on corrected annotations.
   - Analyzes ASR error patterns for continuous improvement.
   - Monitors key metrics to track the performance of the pipeline.

## Project Structure

- **src/**: Contains the source code for the pipeline, organized into modules for ASR, NER, normalization, note generation, feedback, and utilities.
- **data/**: Stores raw, processed, and output audio data.
- **config/**: Contains configuration files for ASR and NER modules, as well as templates for note generation.
- **tests/**: Includes unit and integration tests for the various components of the pipeline.
- **requirements.txt**: Lists the dependencies required for the project.
- **setup.py**: Used for packaging the project and managing dependencies.

## Installation

To install the required dependencies, run:

```
pip install -r updated_requirements.txt
```

## Usage

To run the pipeline, execute the main script:

```
python src\medical-nlp-pipeline\src\api.py
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.