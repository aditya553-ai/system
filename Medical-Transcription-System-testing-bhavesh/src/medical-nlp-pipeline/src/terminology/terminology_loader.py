# Path: /d:/arogo/whisper/medical-transcription-pipeline/src/medical-nlp-pipeline/src/terminology/terminology_loader.py

import csv
import os
import logging

def load_terminology_dictionary(file_path):
    """Load terminology dictionary from CSV file
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        List of terminology entries
    """
    terminology = []
    
    try:
        if not os.path.exists(file_path):
            logging.warning(f"Terminology file not found: {file_path}")
            return []
            
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                terminology.append(row)
                
        return terminology
    except Exception as e:
        logging.error(f"Error loading terminology file {file_path}: {e}")