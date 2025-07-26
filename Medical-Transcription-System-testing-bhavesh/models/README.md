# Medical Transcription Pipeline Models

This document provides an overview of the models used in the Medical Transcription Pipeline project.

## Automatic Speech Recognition (ASR) Model

- **Model Name**: Whisper (small)
- **Purpose**: Converts clinician-patient audio interactions into raw text.
- **Key Features**:
  - Language detection
  - Noise reduction and silence trimming
  - Supports various audio formats

## Named Entity Recognition (NER) Model

- **Model Name**: SciSpacy's en_core_sci_md or John Snow Labs' Spark NLP healthcare pipelines
- **Purpose**: Identifies and tags medical entities in the transcribed text.
- **Entity Categories**:
  - Symptoms (e.g., "shortness of breath")
  - Diagnoses (e.g., "type II diabetes mellitus")
  - Medications (e.g., "metformin 500 mg")
  - Dosages & Frequencies (e.g., "twice daily")
  - Lab Tests (e.g., "HbA1c")

## Normalization Models

- **Terminology Resolver**: Spark NLP's sbiobertresolve_rxnorm_nih for RxNorm, ICD-10-CM resolver
- **Purpose**: Maps raw entity text to standardized medical codes.
- **Multi-Terminology Resolvers**: Aligns entities to SNOMED CT, ICD-10-CM, LOINC, CPT, HCPCS, and RxNorm.

## Feedback and Continuous Improvement

- **Model Retraining**: Periodic fine-tuning of NER and resolver models based on clinician corrections.
- **Error Mitigation**: Fuzzy matching and lexicon-based post-processing to improve ASR accuracy.

## Usage

To utilize these models, ensure that the appropriate configurations are set in the `config` directory and that the necessary dependencies are installed as specified in `requirements.txt`. 

For further details on implementation, refer to the respective modules in the `src` directory.