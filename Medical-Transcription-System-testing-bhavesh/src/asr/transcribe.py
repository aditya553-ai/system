import whisper
import numpy as np
import torch
from typing import Union, Optional
import os

# Global model cache to avoid reloading
_whisper_model = None
_whisper_model_name = None
_whisper_config = None

def transcribe_with_whisper(audio: Union[str, np.ndarray], 
                            model_name: str = "base",
                            language: str = None,
                            task: str = "transcribe",
                            force_reload: bool = False) -> str:
    """
    Transcribe audio using OpenAI's Whisper model
    
    Args:
        audio: Path to audio file or numpy array of audio data
        model_name: Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
        language: Language code (e.g. 'en' for English) or None for auto-detect
        task: 'transcribe' or 'translate' (translate to English)
        force_reload: Force reload the model even if it's already loaded
        
    Returns:
        Transcribed text
    """
    global _whisper_model, _whisper_model_name, _whisper_config
    
    # Create a config hash to determine if we need to reload
    current_config = f"{model_name}_{language}_{task}"
    
    # Load model if not already loaded or if configuration has changed or if forced
    if (_whisper_model is None or 
        _whisper_model_name != model_name or
        _whisper_config != current_config or
        force_reload):
        print(f"Loading Whisper model '{model_name}'...")
        try:
            # Clear previous model from memory if it exists
            if _whisper_model is not None:
                # Try to clear CUDA memory if applicable
                if hasattr(_whisper_model, 'to'):
                    _whisper_model.to('cpu')
                
                # Try to explicitly delete
                del _whisper_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            # Load the new model
            _whisper_model = whisper.load_model(model_name)
            _whisper_model_name = model_name
            _whisper_config = current_config
            print(f"Whisper model '{model_name}' loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return f"Transcription failed: model could not be loaded - {str(e)}"
    else:
        print(f"Using cached Whisper model '{_whisper_model_name}'")
    
    # Transcribe audio
    try:
        # Set appropriate options - using best_of only, not beam_size
        options = {
            "language": language,
            "task": task,
            "best_of": 5,  # Only use best_of, not beam_size
            "fp16": torch.cuda.is_available()
        }
        
        print(f"Transcribing with options: {options}")
        
        # Handle different input types
        if isinstance(audio, str):
            print(f"Audio path received: {audio}")
            audio = os.path.normpath(audio)
            audio = audio.replace("\\", "/")
            print(f"Sanitized audio path: {audio}")

            if not os.path.exists(audio):
                return f"Transcription failed: file {audio} not found"
    
            result = _whisper_model.transcribe(audio, **options)
        else:
            # Audio is a numpy array
            result = _whisper_model.transcribe(audio, **options)
        
        return result["text"].strip()
        
    except Exception as e:
        print(f"Transcription error: {e}")
        return f"Transcription failed: {str(e)}"

def transcribe_audio(audio_data: np.ndarray, force_reload: bool = False) -> str:
    """
    Transcribe audio data to text
    
    Args:
        audio_data: Numpy array of audio samples
        force_reload: Force model reload
        
    Returns:
        Transcribed text
    """
    return transcribe_with_whisper(audio_data, model_name="base", force_reload=force_reload)

# Add a utility function to explicitly reset the model
def reset_whisper_model():
    """Force reset the cached Whisper model"""
    global _whisper_model, _whisper_model_name, _whisper_config
    
    if _whisper_model is not None:
        # Try to clear CUDA memory if applicable
        if hasattr(_whisper_model, 'to'):
            _whisper_model.to('cpu')
        
        # Try to explicitly delete
        del _whisper_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        _whisper_model = None
        _whisper_model_name = None
        _whisper_config = None
        print("Whisper model cache cleared")