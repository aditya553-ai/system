import numpy as np
import librosa
import torch
from resemblyzer import VoiceEncoder, preprocess_wav

def extract_speaker_embeddings(audio_segments, sample_rate=16000, model=None):
    """
    Extract speaker embeddings from audio segments using the Resemblyzer VoiceEncoder
    
    Parameters:
    - audio_segments: List of audio segments or single audio array
    - sample_rate: Sampling rate of the audio
    - model: Optional pre-loaded model (will load Resemblyzer if None)
    
    Returns:
    - numpy array of embeddings
    """
    # Initialize the voice encoder if not provided
    if model is None:
        model = VoiceEncoder()
    
    embeddings = []
    
    # Handle both single audio and list of segments
    if isinstance(audio_segments, list) or (isinstance(audio_segments, np.ndarray) and audio_segments.ndim > 1):
        for segment in audio_segments:
            if len(segment) < 0.1 * sample_rate:  # Skip segments shorter than 0.1s
                continue
            # Preprocess the audio segment for the model
            processed_wav = preprocess_wav(segment, source_sr=sample_rate)
            # Extract embedding
            embedding = model.embed_utterance(processed_wav)
            embeddings.append(embedding)
    else:
        # Process single audio
        processed_wav = preprocess_wav(audio_segments, source_sr=sample_rate)
        embedding = model.embed_utterance(processed_wav)
        embeddings.append(embedding)
    
    return np.array(embeddings)