import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from speaker_embeddings import extract_speaker_embeddings
import librosa

def detect_voice_activity(audio, sample_rate=16000, frame_duration=0.05, threshold=0.01):
    """
    Simple voice activity detection to filter out non-speech segments
    
    Parameters:
    - audio: Audio samples
    - sample_rate: Sample rate of the audio
    - frame_duration: Duration of each frame for VAD in seconds
    - threshold: Energy threshold for speech detection
    
    Returns:
    - Boolean array indicating which frames have speech activity
    """
    frame_length = int(sample_rate * frame_duration)
    # Calculate energy in each frame
    energy = np.array([
        np.mean(audio[i:i+frame_length]**2) 
        for i in range(0, len(audio) - frame_length, frame_length)
    ])
    # Normalize energy
    if len(energy) > 0 and np.max(energy) > 0:
        energy = energy / np.max(energy)
    # Apply threshold for speech detection
    return energy > threshold

def segment_audio_by_speaker(audio, sample_rate=16000, frame_duration=1.0, min_speakers=2, max_speakers=5):
    """
    Segment audio into different speakers with improved accuracy
    
    Parameters:
    - audio: Numpy array of audio samples
    - sample_rate: Sample rate of the audio
    - frame_duration: Duration of each frame in seconds
    - min_speakers: Minimum number of expected speakers
    - max_speakers: Maximum number of expected speakers
    
    Returns:
    - Array of speaker labels for each frame
    """
    frame_length = int(sample_rate * frame_duration)
    num_frames = len(audio) // frame_length
    
    if num_frames < 5:  # Need enough segments for reliable clustering
        print("Warning: Audio too short for reliable diarization")
        return np.zeros(max(1, num_frames), dtype=int)
    
    # Split audio into frames
    frames = np.array([audio[i * frame_length:(i + 1) * frame_length] for i in range(num_frames)])
    
    # Apply voice activity detection to filter out non-speech frames
    vad_frames = []
    vad_indices = []
    
    for i, frame in enumerate(frames):
        # Detect voice activity in this frame
        vad_result = detect_voice_activity(frame, sample_rate)
        speech_percent = np.mean(vad_result) if len(vad_result) > 0 else 0
        
        # Keep frames with significant speech activity (>30% of the frame)
        if speech_percent > 0.3:
            vad_frames.append(frame)
            vad_indices.append(i)
    
    if len(vad_frames) < 5:
        print("Warning: Not enough speech frames for reliable diarization")
        return np.zeros(num_frames, dtype=int)
    
    # Extract speaker embeddings only for frames with speech
    print(f"Extracting speaker embeddings for {len(vad_frames)} speech frames...")
    speaker_embeddings = extract_speaker_embeddings(vad_frames, sample_rate)
    
    # Normalize embeddings for better clustering
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(speaker_embeddings)
    
    # Use silhouette score to estimate the optimal number of speakers
    best_score = -1
    best_n_speakers = 2  # Default to 2 speakers
    best_labels = None
    
    print("Determining optimal number of speakers...")
    
    # Try different numbers of clusters to find the best fit
    for n_speakers in range(min_speakers, min(max_speakers + 1, len(vad_frames) // 3)):
        # Try multiple initializations to avoid local minima
        for seed in [42, 101, 123]:
            clustering = KMeans(n_clusters=n_speakers, random_state=seed, n_init=10)
            labels = clustering.fit_predict(scaled_embeddings)
            
            # Skip if any cluster has too few samples
            counts = np.bincount(labels)
            if np.any(counts < 3):  # Require at least 3 segments per speaker
                continue
            
            # Calculate silhouette score (measure of cluster quality)
            if len(np.unique(labels)) > 1:  # Silhouette requires at least 2 clusters
                try:
                    score = silhouette_score(scaled_embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_n_speakers = n_speakers
                        best_labels = labels
                except:
                    continue
    
    # If no good clustering found, try hierarchical clustering
    if best_labels is None or best_score < 0.1:
        print("KMeans clustering failed, trying hierarchical clustering...")
        try:
            # Use hierarchical clustering instead
            clustering = AgglomerativeClustering(
                n_clusters=min_speakers,  # Start with minimum speakers
                linkage='ward'  # Ward linkage works well for voice data
            )
            best_labels = clustering.fit_predict(scaled_embeddings)
            best_n_speakers = len(np.unique(best_labels))
        except Exception as e:
            print(f"Hierarchical clustering failed: {e}. Defaulting to {min_speakers} speakers.")
            # Default to minimum number of speakers
            clustering = KMeans(n_clusters=min_speakers, random_state=42)
            best_labels = clustering.fit_predict(scaled_embeddings)
            best_n_speakers = min_speakers
    
    # Post-processing to consolidate speakers
    # Merge speakers that appear very briefly
    speaker_counts = np.bincount(best_labels)
    frequent_speakers = np.where(speaker_counts >= max(3, len(vad_frames) * 0.1))[0]
    
    if len(frequent_speakers) < best_n_speakers:
        # Remap infrequent speakers to the closest frequent speaker
        remapped_labels = best_labels.copy()
        for i, label in enumerate(best_labels):
            if label not in frequent_speakers:
                # Find closest frequent speaker centroid
                closest_frequent = frequent_speakers[0]
                for freq_spk in frequent_speakers:
                    if np.mean(scaled_embeddings[best_labels == freq_spk], axis=0) @ scaled_embeddings[i] > \
                       np.mean(scaled_embeddings[best_labels == closest_frequent], axis=0) @ scaled_embeddings[i]:
                        closest_frequent = freq_spk
                remapped_labels[i] = closest_frequent
        
        best_labels = remapped_labels
        best_n_speakers = len(frequent_speakers)
    
    print(f"Found {best_n_speakers} speakers with confidence score: {best_score:.3f}")
    
    # Map the labels back to the original frames
    full_labels = np.zeros(num_frames, dtype=int)
    for i, vad_idx in enumerate(vad_indices):
        full_labels[vad_idx] = best_labels[i]
    
    # Fill in labels for non-speech frames using nearest neighbor interpolation
    for i in range(num_frames):
        if i not in vad_indices:
            # Find nearest speech frame
            nearest_idx = min(vad_indices, key=lambda x: abs(x - i))
            full_labels[i] = full_labels[nearest_idx]
    
    # Apply temporal smoothing to reduce speaker switching
    smoothed_labels = smooth_labels(full_labels, window_size=3)
    
    return smoothed_labels

def smooth_labels(labels, window_size=3):
    """
    Apply temporal smoothing to labels to reduce rapid speaker changes
    """
    smoothed = labels.copy()
    half_window = window_size // 2
    
    for i in range(len(labels)):
        if i <= half_window or i >= len(labels) - half_window:
            continue
            
        # Get the window around position i
        window = labels[i-half_window:i+half_window+1]
        
        # Find most common label in the window
        counts = np.bincount(window)
        smoothed[i] = np.argmax(counts)
    
    return smoothed

def diarize_audio(audio, sample_rate=16000):
    """
    Perform speaker diarization on audio data
    
    Parameters:
    - audio: Preprocessed audio numpy array
    - sample_rate: Sample rate of the audio
    
    Returns:
    - Array of speaker labels
    """
    # Get prior knowledge about expected speakers
    expected_speakers = 2  # Default expectation
    
    # Allow override from environment
    import os
    try:
        env_speakers = os.environ.get("EXPECTED_SPEAKERS")
        if env_speakers and env_speakers.isdigit():
            expected_speakers = int(env_speakers)
    except:
        pass
        
    # Use expected speakers to guide diarization
    min_speakers = max(2, expected_speakers - 1)
    max_speakers = expected_speakers + 1  # Reduced from +2 to +1
    
    return segment_audio_by_speaker(
        audio, sample_rate, 
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )