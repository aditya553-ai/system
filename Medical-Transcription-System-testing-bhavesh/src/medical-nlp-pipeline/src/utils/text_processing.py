from typing import List, Dict, Any
import re
import json

class TextProcessor:
    """
    A class for processing text data in the medical NLP pipeline.
    """

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizes the input text into words.
        
        Args:
            text: The input text to tokenize.
        
        Returns:
            A list of tokens.
        """
        return re.findall(r'\b\w+\b', text.lower())

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalizes the input text by removing unwanted characters and lowercasing.
        
        Args:
            text: The input text to normalize.
        
        Returns:
            The normalized text.
        """
        return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """
        Loads a JSON file and returns its content.
        
        Args:
            file_path: The path to the JSON file.
        
        Returns:
            The content of the JSON file as a dictionary.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str) -> None:
        """
        Saves a dictionary to a JSON file.
        
        Args:
            data: The data to save.
            file_path: The path to the JSON file.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)