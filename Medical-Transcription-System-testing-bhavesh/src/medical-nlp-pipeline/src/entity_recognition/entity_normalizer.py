from typing import Dict, List, Any, Optional
import re
import string

class EntityNormalizer:
    """
    Class for normalizing medical entities before mapping to standardized terminologies.
    Handles common transcription errors, abbreviations, and text normalization.
    """
    
    def __init__(self):
        """Initialize the entity normalizer with abbreviations and common corrections"""
        # Common medical abbreviations
        self.abbreviations = {
            "a1c": "hemoglobin a1c",
            "afib": "atrial fibrillation",
            "bp": "blood pressure",
            "bpm": "beats per minute",
            "chf": "congestive heart failure",
            "copd": "chronic obstructive pulmonary disease",
            "cxr": "chest x-ray",
            "dm": "diabetes mellitus",
            "dv": "diabetic",
            "dvt": "deep vein thrombosis",
            "echo": "echocardiogram",
            "eeg": "electroencephalogram",
            "ekg": "electrocardiogram",
            "gerd": "gastroesophageal reflux disease",
            "hpi": "history of present illness",
            "hr": "heart rate",
            "htn": "hypertension",
            "mi": "myocardial infarction",
            "mrsa": "methicillin-resistant staphylococcus aureus",
            "npo": "nothing by mouth",
            "nsaid": "nonsteroidal anti-inflammatory drug",
            "pe": "pulmonary embolism",
            "pmh": "past medical history",
            "pt": "patient",
            "sob": "shortness of breath",
            "tid": "three times a day",
            "uti": "urinary tract infection",
            "vs": "vital signs"
        }
        
        # Common transcription errors and their corrections
        self.common_errors = {
            "metfomin": "metformin",
            "metphormin": "metformin",
            "glucophage": "metformin",
            "lipator": "lipitor",
            "atorvastatin": "atorvastatin",
            "amiodaron": "amiodarone",
            "amoxicilin": "amoxicillin",
            "metropolol": "metoprolol",
            "ibprophin": "ibuprofen",
            "ibprofin": "ibuprofen",
            "arthritus": "arthritis",
            "diabetis": "diabetes",
            "diabeties": "diabetes",
            "hyptertension": "hypertension",
            "hipertension": "hypertension",
            "hiperlipidemia": "hyperlipidemia",
            "colesterol": "cholesterol"
        }
        
        # Compile word boundary regex patterns for efficient matching
        self.abbr_pattern = re.compile(r'\b(' + '|'.join(re.escape(abbr) for abbr in self.abbreviations.keys()) + r')\b', re.IGNORECASE)
        self.error_pattern = re.compile(r'\b(' + '|'.join(re.escape(err) for err in self.common_errors.keys()) + r')\b', re.IGNORECASE)

    def normalize(self, entity_text: str, entity_type: Optional[str] = None) -> str:
        """
        Normalize entity text by fixing common errors, expanding abbreviations,
        and standardizing format.
        
        Args:
            entity_text: The text of the entity to normalize
            entity_type: Optional entity type for type-specific normalizations
            
        Returns:
            Normalized entity text
        """
        if not entity_text:
            return entity_text
            
        # Convert to lowercase for consistent processing
        text = entity_text.lower()
        
        # Remove punctuation that might affect matching
        text = text.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
        
        # Fix common transcription errors
        text = self.error_pattern.sub(lambda m: self.common_errors[m.group(0).lower()], text)
        
        # Expand abbreviations
        text = self.abbr_pattern.sub(lambda m: self.abbreviations[m.group(0).lower()], text)
        
        # Apply type-specific normalizations
        if entity_type:
            text = self._apply_type_specific_normalization(text, entity_type)
        
        return text
    
    def _apply_type_specific_normalization(self, text: str, entity_type: str) -> str:
        """
        Apply normalization rules specific to entity type.
        
        Args:
            text: The text to normalize
            entity_type: The type of entity
            
        Returns:
            Normalized text based on entity type
        """
        if entity_type in ["MEDICATION", "DRUG"]:
            # Remove common medication suffixes for normalization
            for suffix in [" tablet", " capsule", " pill", " injection", " mg", " mcg", " ml"]:
                if text.endswith(suffix):
                    text = text[:-len(suffix)]
            
        elif entity_type in ["LAB_TEST", "TEST"]:
            # Standardize lab test names
            if "a1c" in text:
                text = "hemoglobin a1c"
            elif "glucose" in text and any(word in text for word in ["fasting", "fast"]):
                text = "fasting blood glucose"
            
        elif entity_type in ["DOSAGE"]:
            # Standardize dosage formats
            text = re.sub(r'(\d+)\s*mg', r'\1 mg', text)  # Ensure space between number and unit
            text = re.sub(r'(\d+)\s*mcg', r'\1 mcg', text)
            text = re.sub(r'(\d+)\s*ml', r'\1 ml', text)
            
        return text

    def normalize_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize a list of entities.
        
        Args:
            entities: List of entities to normalize
            
        Returns:
            List of normalized entities
        """
        normalized_entities = []
        
        for entity in entities:
            # Create a copy to avoid modifying original
            normalized_entity = entity.copy()
            
            # Keep original text
            normalized_entity["original_text"] = entity["text"]
            
            # Normalize the text
            normalized_entity["text"] = self.normalize(entity["text"], entity.get("label"))
            
            # Add to normalized list
            normalized_entities.append(normalized_entity)
            
        return normalized_entities