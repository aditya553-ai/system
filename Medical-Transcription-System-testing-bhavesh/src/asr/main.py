import os
import numpy as np
import time
from preprocessing import preprocess_audio
from diarization import diarize_audio
from transcribe import transcribe_audio, transcribe_with_whisper, reset_whisper_model
import json
from datetime import datetime
import torch
import sys

def main(audio_file_path):
    reset_whisper_model()
    print(f"Processing audio file: {audio_file_path}")
    
    # Check if file exists
    if not os.path.exists(audio_file_path):
        print(f"Error: File {audio_file_path} does not exist")
        return
    
    # Create output folder for transcriptions if it doesn't exist
    output_dir = os.path.join(os.path.dirname(audio_file_path), "transcriptions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename based on input filename
    base_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{base_filename}_{timestamp}.txt")
    output_json = os.path.join(output_dir, f"{base_filename}_{timestamp}.json")
    
    # Step 1: Preprocess the audio
    print("Step 1: Preprocessing audio...")
    preprocessed_audio = preprocess_audio(audio_file_path)
    print(f"Audio preprocessing complete. Shape: {preprocessed_audio.shape}")
    
    # Step 2: Diarize the audio (identify speakers)
    print("Step 2: Performing speaker diarization...")
    speaker_labels = diarize_audio(preprocessed_audio)
    unique_speakers = np.unique(speaker_labels)
    print(f"Identified {len(unique_speakers)} speakers")
    
    # Step 3: Identify speaker turns and transcribe each turn
    print("Step 3: Identifying speaker turns...")
    frame_duration = 1.0  # Same as in diarization
    sample_rate = 16000   # Default sample rate
    frame_length = int(sample_rate * frame_duration)
    
    # Find speaker turns (where the speaker changes)
    speaker_turns = []
    current_speaker = speaker_labels[0]
    turn_start = 0
    min_turn_duration = 2  # Minimum turn duration in frames (seconds)
    
    # Find all speaker transitions with filtering for minimum duration
    for i in range(1, len(speaker_labels)):
        if speaker_labels[i] != current_speaker:
            # Potential speaker change - check if current segment is long enough
            turn_duration = i - turn_start
            
            if turn_duration >= min_turn_duration:
                # Current segment is long enough, save it
                speaker_turns.append({
                    "speaker": int(current_speaker),
                    "start_frame": turn_start,
                    "end_frame": i - 1,
                    "start_time": turn_start * frame_duration,
                    "end_time": i * frame_duration
                })
                # Start new segment
                turn_start = i
                current_speaker = speaker_labels[i]
            else:
                # Current segment too short, just continue with the previous speaker
                # This effectively ignores very brief interruptions
                speaker_labels[i] = current_speaker
    
    # Add the final turn if it's long enough
    final_turn_duration = len(speaker_labels) - turn_start
    if final_turn_duration >= min_turn_duration:
        speaker_turns.append({
            "speaker": int(current_speaker),
            "start_frame": turn_start,
            "end_frame": len(speaker_labels) - 1,
            "start_time": turn_start * frame_duration,
            "end_time": len(speaker_labels) * frame_duration
        })
    
    print(f"Found {len(speaker_turns)} speaker turns in the conversation")
    
    # Store results in a structured format
    transcription_data = {
        "metadata": {
            "filename": audio_file_path,
            "timestamp": timestamp,
            "num_speakers": len(unique_speakers),
            "duration_seconds": len(preprocessed_audio) / sample_rate
        },
        "speaker_turns": []
    }
    
    # Open the output file for writing
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Transcript of {audio_file_path}\n")
        f.write(f"Processed on: {timestamp}\n")
        f.write(f"Number of speakers detected: {len(unique_speakers)}\n")
        f.write(f"Total duration: {len(preprocessed_audio) / sample_rate:.2f} seconds\n\n")
        
        # Process each speaker turn one at a time
        for turn_idx, turn in enumerate(speaker_turns):
            # Extract audio for this turn
            start_idx = turn["start_frame"] * frame_length
            end_idx = min((turn["end_frame"] + 1) * frame_length, len(preprocessed_audio))
            turn_audio = preprocessed_audio[start_idx:end_idx]
            
            # Skip very short turns
            if len(turn_audio) < sample_rate * 0.5:  # Skip turns shorter than 0.5 seconds
                continue
                
            # Process this speaker turn
            print(f"\nProcessing turn {turn_idx+1}/{len(speaker_turns)}: Speaker {turn['speaker']} " +
                  f"({turn['start_time']:.1f}s - {turn['end_time']:.1f}s)")
            
            # Transcribe this turn using Whisper
            try:
                turn_text = transcribe_audio(turn_audio)
                if not isinstance(turn_text, str):
                    turn_text = str(turn_text)
                turn_text = turn_text.strip()
            except Exception as e:
                print(f"Transcription error: {e}")
                turn_text = ""
            
            # Filter out likely noise or meaningless utterances
            meaningful_content = True
            noise_words = ["yeah", "yah", "bye", "um", "uh", "ah", "oh", "hm", "hmm", "eh"]
            
            # Check if the text is just noise words
            if turn_text.lower() in noise_words or len(turn_text.split()) <= 1:
                meaningful_content = False
                print(f"Skipping likely noise: '{turn_text}'")
            
            # If we got meaningful text
            if turn_text and meaningful_content:
                # Write to file immediately
                turn_info = f"[Speaker {turn['speaker']}] ({turn['start_time']:.1f}s - {turn['end_time']:.1f}s): {turn_text}\n"
                f.write(turn_info)
                f.flush()  # Ensure it's written immediately
                
                # Print to console in real-time
                print(turn_info.strip())
                
                # Store for JSON output
                turn_data = {
                    "speaker": int(turn["speaker"]),
                    "start_time": turn["start_time"],
                    "end_time": turn["end_time"],
                    "text": turn_text
                }
                transcription_data["speaker_turns"].append(turn_data)
        
        f.write("\n--- End of Transcript ---\n")
    
    # Save the structured data as JSON
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(transcription_data, json_file, indent=2)
    
    print(f"\nTranscription complete! Output saved to:")
    print(f"- Text file: {output_file}")
    print(f"- JSON file: {output_json}")
    
    # Verify files were written correctly
    try:
        with open(output_file, "r", encoding="utf-8") as check_file:
            content = check_file.read()
            if len(content.split("\n")) < 5:  # Just a basic check
                print("Warning: Transcript file seems too short. There may have been an issue.")
                
        with open(output_json, "r", encoding="utf-8") as check_json:
            json_content = json.load(check_json)
            if len(json_content.get("speaker_turns", [])) == 0:
                print("Warning: JSON file has no transcription content. There may have been an issue.")
    except Exception as e:
        print(f"Error verifying output files: {e}")
    
    return transcription_data

if __name__ == "__main__":
    # Allow command line argument for audio file path
    audio_file_path = "D:\\arogo\\whisper\\medical-transcription-pipeline\\src\\asr\\data\\audiofinal2.mp3"
    
    # If a command line argument is provided, use it as the audio file path
    if len(sys.argv) > 1:
        audio_file_path = sys.argv[1]
    
    # Check if you have a valid audio file
    try:
        from pydub import AudioSegment
        print("Opening file to verify it's valid...")
        audio = AudioSegment.from_file(audio_file_path)
        print(f"File is valid: {len(audio)/1000} seconds, {audio.channels} channels, {audio.frame_rate} Hz")
    except Exception as e:
        print(f"Error verifying file: {e}")
        print("Please ensure you have a valid audio file and ffmpeg is properly installed")
        exit(1)
        
    main(audio_file_path)