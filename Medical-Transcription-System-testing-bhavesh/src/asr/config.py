import os

# Configuration settings for the speech processing project

# File paths
RAW_AUDIO_PATH = os.path.join('data', 'raw')
PROCESSED_AUDIO_PATH = os.path.join('data', 'processed')

# Model parameters
SAMPLE_RATE = 16000
NOISE_REDUCTION_THRESHOLD = 0.01
SILENCE_TRIMMING_THRESHOLD = 0.01
FRAME_DURATION = 1.0  # in seconds

# Speaker embeddings settings
EMBEDDING_MODEL_PATH = 'path/to/speaker_embedding_model'  # Update with actual model path
EMBEDDING_DIMENSION = 128  # Example dimension size for embeddings

# Whisper model settings
WHISPER_MODEL_TYPE = "small"  # Model size for Whisper
