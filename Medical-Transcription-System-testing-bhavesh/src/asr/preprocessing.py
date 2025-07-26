import numpy as np
import librosa
from speaker_embeddings import extract_speaker_embeddings
from pydub import AudioSegment
import os
import io

def load_and_preprocess_audio(file_path, sr=16000):
    """Load audio from file path and preprocess it"""
    # Check if file_path is already a numpy array
    if isinstance(file_path, np.ndarray):
        # File path is already audio data
        print("Input is already a numpy array, skipping loading")
        return file_path
        
    try:
        # Try librosa first
        audio, _ = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"Librosa loading failed: {e}, trying pydub instead")
        # Fallback to pydub
        audio = load_audio_with_pydub(file_path, sr)
    return audio

def load_audio_with_pydub(file_path, sr=16000):
    """Load audio using pydub as fallback"""
    if isinstance(file_path, np.ndarray):
        # File path is already audio data
        return file_path
        
    audio = AudioSegment.from_file(file_path)
    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)
    # Set sample rate
    if audio.frame_rate != sr:
        audio = audio.set_frame_rate(sr)
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    # Normalize to [-1.0, 1.0]
    return samples / (1 << (8 * audio.sample_width - 1))

def noise_reduction(audio):
    # Implement noise reduction logic here
    return audio

def silence_trimming(audio, threshold=0.01):
    non_silent_indices = np.where(np.abs(audio) > threshold)[0]
    if len(non_silent_indices) == 0:
        return audio  # Return original if no non-silent parts found
    return audio[non_silent_indices[0]:non_silent_indices[-1] + 1]

def preprocess_audio(file_path):
    """Main preprocessing function that applies all preprocessing steps"""
    audio = load_and_preprocess_audio(file_path)
    audio = noise_reduction(audio)
    audio = silence_trimming(audio)
    return audio