import spacy
import json
import os
import re
from typing import Dict, List, Any
import sys

class MedicalEntityRecognizer:
    """
    Class for recognizing medical entities in transcripts using
    spaCy NER enhanced with medical term dictionaries
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the medical entity recognizer
        
        Args:
            model_name: Name of the spaCy model to use
        """
        # Try to load the spaCy model, download it if not available
        try:
            print(f"Loading spaCy model '{model_name}'...")
            self.nlp = spacy.load(model_name)
        except IOError:
            print(f"Model '{model_name}' not found. Attempting to download...")
            try:
                # Attempt to download the model
                import subprocess
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
                self.nlp = spacy.load(model_name)
                print(f"Successfully downloaded and loaded '{model_name}'")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("Falling back to basic spaCy model")
                # Create a blank model as fallback
                self.nlp = spacy.blank("en")
        
        # Initialize dictionaries and patterns for medical entity recognition
        self._initialize_patterns()

    # Add this method to your MedicalEntityRecognizer class
    def extract_entities(self, text):
        """Extract medical entities from text"""
        if not text:
            return []
            
        entities = []
        
        try:
            # Use spaCy to get base entities with error handling around entity_linker
            doc = None
            try:
                doc = self.nlp(text)
            except ValueError as e:
                if "Knowledge base for component 'entity_linker' is empty" in str(e):
                    # Recreate the nlp pipeline without entity_linker
                    print("Recreating spaCy pipeline without entity_linker")
                    self.nlp = spacy.load(self.nlp.meta["name"], disable=["entity_linker"])
                    doc = self.nlp(text)
                else:
                    raise  # Re-raise other ValueError exceptions
                    
            # Process spaCy entities
            for ent in doc.ents:
                entity = {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": "spacy"
                }
                entities.append(entity)
                
            # Extract additional entities using regex patterns
            for entity_type, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    for match in pattern.finditer(text):
                        matched_text = match.group(1) if match.groups() else match.group(0)
                        start = match.start()
                        end = match.end()
                        
                        # Check for overlap with existing entities
                        overlap = False
                        for e in entities:
                            if (start >= e["start"] and start < e["end"]) or \
                            (end > e["start"] and end <= e["end"]):
                                overlap = True
                                break
                                
                        if not overlap:
                            entity = {
                                "text": matched_text,
                                "label": entity_type,
                                "start": start,
                                "end": end,
                                "source": "pattern"
                            }
                            entities.append(entity)
            
            return entities
        
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return []
    
    def _initialize_patterns(self):
        """Initialize patterns for medical entity recognition"""
        # Define entity types and common patterns
        self.entity_types = [
            "MEDICATION", "SYMPTOM", "CONDITION", "TREATMENT", "TEST", "ANATOMY",
            "PROCEDURE", "TEMPORAL", "DOSAGE", "FREQUENCY"
        ]
        
        # Add medical entity patterns
        self.medication_patterns = [
            r"\b(?:taking|on|prescribed|using|dose of|medication|drug|pill|capsule|tablet|injection)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"\b([A-Z][a-z]+(?:in|ol|ide|ine|one|ate|zole|ium|oxin|arin|icin|mycin|dryl))\b"
        ]
        
        self.symptom_patterns = [
            r"\b(?:experiencing|having|feeling|suffering from|complaining of)\s+([a-z]+(?:\s+[a-z]+){0,4})",
            r"\b(pain|ache|discomfort|fever|cough|nausea|dizziness|fatigue|weakness|shortness of breath|difficulty breathing)\b"
        ]
        
        self.condition_patterns = [
            r"\b(?:diagnosed with|suffers from|history of|chronic|acute)\s+([a-z]+(?:\s+[a-z]+){0,4})",
            r"\b(diabetes|hypertension|asthma|arthritis|depression|anxiety|COPD|CHF|CAD)\b"
        ]
        
        self.temporal_patterns = [
            r"\b(for \d+ (?:day|week|month|year)s?)\b",
            r"\b(since \d+)\b",
            r"\b(last (?:day|week|month|year))\b",
            r"\b(\d+ times? a (?:day|week|month))\b",
            r"\b(every \d+ hours?)\b",
            r"\b(once|twice|three times) a (?:day|week|month)\b"
        ]
        
        # Compile regex patterns for faster matching
        self.compiled_patterns = {}
        for pattern_type, patterns in [
            ("MEDICATION", self.medication_patterns),
            ("SYMPTOM", self.symptom_patterns),
            ("CONDITION", self.condition_patterns),
            ("TEMPORAL", self.temporal_patterns)
        ]:
            self.compiled_patterns[pattern_type] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def _load_medical_dictionaries(self):
        """Load medical term dictionaries for enhanced entity recognition"""
        # Dictionary of common medical terms by category
        med_dict = {
            "MEDICATION": [
                "metformin", "insulin", "glipizide", "glyburide", "sitagliptin",
                "lisinopril", "atorvastatin", "simvastatin", "amlodipine", "metoprolol"
            ],
            "CONDITION": [
                "diabetes", "hypertension", "hyperlipidemia", "diabetic peripheral neuropathy", 
                "peripheral neuropathy", "neuropathy", "hypoglycemia", "hyperglycemia"
            ],
            "SYMPTOM": [
                "pain", "burning pain", "tingling", "numbness", "dizziness", "sweating", 
                "rapid heartbeat", "heart races", "sweaty", "lightheaded", "dizzy"
            ],
            "ANATOMY": [
                "feet", "legs", "toes", "extremities", "lower extremities"
            ]
        }
        
        # Try to load custom dictionaries from file if available
        dict_path = os.path.join(os.path.dirname(__file__), "data", "medical_dictionaries.json")
        try:
            if os.path.exists(dict_path):
                with open(dict_path, 'r', encoding='utf-8') as f:
                    custom_dict = json.load(f)
                    # Merge with default dictionaries
                    for category, terms in custom_dict.items():
                        if category in med_dict:
                            med_dict[category].extend(terms)
                        else:
                            med_dict[category] = terms
                    print(f"Loaded custom medical dictionaries from {dict_path}")
        except Exception as e:
            print(f"Warning: Could not load custom dictionaries: {e}")
        
        return med_dict
    
    def _extract_entities(self, doc):
        """Extract entities from a spaCy document"""
        entities = []
        
        # Get entities from spaCy NER
        for ent in doc.ents:
            # Skip very short entities that are likely noise
            if len(ent.text) < 2:
                continue
                
            entity = {
                "text": ent.text,
                "label": ent.label_,
                "confidence": 1.0  # spaCy doesn't provide confidence scores
            }
            entities.append(entity)
        
        # Enhance with medical dictionary lookup
        self._enhance_with_medical_terms(doc.text, entities)
        
        return entities
    
    def _enhance_with_medical_terms(self, text, entities):
        """
        Enhance entity recognition with medical dictionary lookup
        
        Args:
            text: The text to analyze
            entities: List of existing entities to enhance
        """
        text_lower = text.lower()
        found_spans = []  # Track spans to avoid duplicates
        
        # Extract spans of existing entities to avoid overlap
        existing_spans = []
        for entity in entities:
            entity_text = entity["text"].lower()
            start = text_lower.find(entity_text)
            if start >= 0:
                end = start + len(entity_text)
                existing_spans.append((start, end))
        
        # Check for medical terms
        for category, terms in self.med_dictionaries.items():
            for term in terms:
                term_lower = term.lower()
                
                # Find all occurrences
                for match in re.finditer(r'\b' + re.escape(term_lower) + r'\b', text_lower):
                    start, end = match.span()
                    
                    # Check if this span overlaps with an existing entity
                    overlap = False
                    for e_start, e_end in existing_spans:
                        if (start >= e_start and start < e_end) or (end > e_start and end <= e_end):
                            overlap = True
                            break
                    
                    if not overlap and (start, end) not in found_spans:
                        # Convert "MEDICATION" category to "DRUG" for compatibility
                        label = "DRUG" if category == "MEDICATION" else category
                        
                        # Add the term as a new entity
                        entities.append({
                            "text": text[start:end],
                            "label": label,
                            "confidence": 0.9,  # Dictionary-based match
                            "source": "medical_dictionary"
                        })
                        found_spans.append((start, end))
                        
                        # Also track this span to avoid future overlaps
                        existing_spans.append((start, end))
    
    def process_transcript(self, transcript_data_or_path):
        """
        Process a transcript JSON file or object for entity recognition
        
        Args:
            transcript_data_or_path: Path to transcript JSON file or JSON object
            
        Returns:
            Updated transcript with recognized entities
        """
        # Load transcript from file if a path is provided
        if isinstance(transcript_data_or_path, str):
            with open(transcript_data_or_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
        else:
            transcript_data = transcript_data_or_path
            
        # Create a copy to store results
        result = transcript_data.copy()
        
        # Process each speaker turn
        for turn in result.get("speaker_turns", []):
            text = turn.get("text", "")
            if text:
                doc = self.nlp(text)
                entities = self._extract_entities(doc)
                turn["entities"] = entities
        
        # Calculate entity statistics
        entity_counts = {}
        total_entities = 0
        
        for turn in result.get("speaker_turns", []):
            for entity in turn.get("entities", []):
                label = entity.get("label", "UNKNOWN")
                if label not in entity_counts:
                    entity_counts[label] = 0
                entity_counts[label] += 1
                total_entities += 1
        
        # Add entity statistics
        result["entity_stats"] = {
            "total_entities": total_entities,
            "entity_counts_by_type": entity_counts
        }
        
        return result
    
    def process_transcript_optimized(self, transcript_data):
        """
        Process a transcript JSON object with optimized batching for speed
        
        Args:
            transcript_data: Transcript data as a JSON object
            
        Returns:
            Updated transcript with recognized entities
        """
        result = transcript_data.copy()
        
        # Batch process all texts at once for efficiency
        batch_texts = []
        turn_indices = []
        
        # Collect all texts for batch processing
        for i, turn in enumerate(transcript_data.get("speaker_turns", [])):
            text = turn.get("text", "")
            if text:
                batch_texts.append(text)
                turn_indices.append(i)
        
        # Batch process all texts
        if batch_texts:
            try:
                # Process texts in batches for better performance
                batch_size = 10
                all_entities = []
                
                for i in range(0, len(batch_texts), batch_size):
                    batch = batch_texts[i:i+batch_size]
                    # Use spaCy's pipe for faster batch processing
                    docs = list(self.nlp.pipe(batch))
                    batch_entities = [self._extract_entities(doc) for doc in docs]
                    all_entities.extend(batch_entities)
                
                # Update turns with entity information
                for idx, entities, i in zip(turn_indices, all_entities, range(len(turn_indices))):
                    result["speaker_turns"][idx]["entities"] = entities
            
            except Exception as e:
                print(f"Error in batch entity extraction: {e}")
                # Fallback to individual processing
                for i, turn in enumerate(result.get("speaker_turns", [])):
                    text = turn.get("text", "")
                    if text:
                        try:
                            doc = self.nlp(text)
                            entities = self._extract_entities(doc)
                            turn["entities"] = entities
                        except Exception as e2:
                            print(f"Error processing turn {i}: {e2}")
                            turn["entities"] = []
        
        # Calculate entity statistics
        entity_counts = {}
        total_entities = 0
        
        for turn in result.get("speaker_turns", []):
            for entity in turn.get("entities", []):
                label = entity.get("label", "UNKNOWN")
                if label not in entity_counts:
                    entity_counts[label] = 0
                entity_counts[label] += 1
                total_entities += 1
        
        # Add entity statistics
        result["entity_stats"] = {
            "total_entities": total_entities,
            "entity_counts_by_type": entity_counts
        }
        
        # Post-process to ensure "Metformin" is recognized as a medication, not an organization
        self._post_process_entities(result)
        
        return result
    
    def _post_process_entities(self, data):
        """Post-process entities to fix common recognition errors"""
        medication_names = set(name.lower() for name in self.med_dictionaries.get("MEDICATION", []))
        
        for turn in data.get("speaker_turns", []):
            for entity in turn.get("entities", []):
                # Fix common medication misclassification
                if entity.get("label") == "ORG" and entity.get("text", "").lower() in medication_names:
                    entity["label"] = "DRUG"
                    entity["type"] = "medication"
                    entity["source"] = "post_processed"