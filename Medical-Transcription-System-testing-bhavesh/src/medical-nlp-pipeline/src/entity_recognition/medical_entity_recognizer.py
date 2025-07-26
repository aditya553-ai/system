from typing import Dict, List, Any
import spacy
import re
import json
import os
import sys
import subprocess
import logging
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from entity_recognition.base_recognizer import BaseRecognizer
from terminology.terminology_resolver import TerminologyResolver
from llm.llm_client import LLMClient
from model_manager import ModelManager


class MedicalEntityRecognizer(BaseRecognizer):
    """
    Class for recognizing medical entities in transcripts using
    spaCy NER enhanced with medical term dictionaries and hybrid mapping.
    """
    
    def __init__(self, model_manager_instance: ModelManager, model_name: str = "en_core_web_sm"):
        """
        Initialize the medical entity recognizer
        
        Args:
            model_name: Name of the spaCy model to use
        """
        super().__init__()

        self.model_manager = model_manager_instance

        self.entity_types = [
            "MEDICATION", "SYMPTOM", "CONDITION", "TREATMENT", "TEST", "ANATOMY",
            "PROCEDURE", "TEMPORAL", "DOSAGE", "FREQUENCY"
        ]
        
        # Get spaCy model from model manager
        self.nlp = self.model_manager.get_model("spacy")
        if not self.nlp:
            logging.warning(f"{self.__class__.__name__}: spaCy model not found via ModelManager. Attempting direct load of 'en_core_web_sm'.")
            try:
                import spacy # Local import for fallback is fine here
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logging.error(f"{self.__class__.__name__}: Failed to load spaCy model 'en_core_web_sm' directly: {e}")
                self.nlp = None # Ensure self.nlp is defined even on failure
        
        try:
            self.terminology_resolver = TerminologyResolver(model_manager_instance=self.model_manager)
        except Exception as e:
            logging.error(f"Error initializing TerminologyResolver in MedicalEntityRecognizer: {e}", exc_info=True)
            raise # Re-raise to indicate a critical failure

        try:
            self.llm_client = LLMClient(model_manager_instance=self.model_manager)
        except Exception as e:
            logging.error(f"Error initializing LLMClient in MedicalEntityRecognizer: {e}", exc_info=True)
            raise # Re-raise to indicate a critical failure

        logging.debug(f"{self.__class__.__name__} base components initialized. Initializing dictionaries and patterns...")
        
        self._initialize_medication_dict() # Check this method for bare 'model_manager' usage
        self._initialize_patterns()      # Check this method for bare 'model_manager' usage
        
        logging.info(f"{self.__class__.__name__} initialized successfully.")

    def _initialize_medication_dict(self):
        """Initialize dictionary of common medications"""
        self.medication_dict = {
            # Common pain medications 
            "aspirin": True, "acetaminophen": True, "tylenol": True, "ibuprofen": True, 
            "advil": True, "motrin": True, "naproxen": True, "aleve": True,
            "celebrex": True, "celecoxib": True, "diclofenac": True, "meloxicam": True,
            "excedrin": True, "vicodin": True, "percocet": True, "oxycodone": True,
            "hydrocodone": True, "codeine": True, "morphine": True, "tramadol": True,
            
            # Common migraine medications
            "sumatriptan": True, "imitrex": True, "rizatriptan": True, "maxalt": True,
            "zolmitriptan": True, "zomig": True, "almotriptan": True, "axert": True,
            "eletriptan": True, "relpax": True, "frovatriptan": True, "frova": True,
            "naratriptan": True, "amerge": True, "ergotamine": True, "cafergot": True,
            "propranolol": True, "inderal": True, "timolol": True, "topiramate": True,
            "topamax": True, "valproate": True, "depakote": True, "botox": True,
            
            # Common non-medications that might be misclassified
            "pain": False, "relief": False, "for relief": False, "hydrate": False,
            "water": False, "rest": False, "sleep": False, "exercise": False,
            "caffeine": False, "coffee": False, "tea": False, "alcohol": False,
            "stress": False, "food": False, "diet": False, "meal": False,
            "sugar": False, "salt": False, "vitamin": False, "supplement": False,
        }

    def _apply_rules(self, text: str) -> List[Dict[str, Any]]:
        """Apply rule-based pattern matching to identify medical entities"""
        entities = []
        
        # Apply medication patterns
        medication_patterns = [
            r"(\b[A-Z][a-z]+(?:mab|nib|zumab)\b)",  # Monoclonal antibodies and kinase inhibitors
            r"(\b\w+(?:cillin|mycin|oxacin|cycline)\b)",  # Antibiotics
            r"(\b\w+(?:sartan|pril|olol|dipine)\b)",  # CV medications
            r"(\b\w+statin\b)",  # Cholesterol medications
            r"(\bprescribe\s+([A-Z][a-z]+(?:an|in|ol|ide|ine|one|ate|zole|ium|oxin|arin|icin|mycin|dryl))\b)",  # Words after "prescribe" with medication endings
            r"(\b(sumatriptan|ondansetron|ibuprofen|acetaminophen|metformin|lisinopril|atorvastatin|simvastatin|amlodipine|metoprolol|aspirin|tylenol)\b)"  # Common medications by name
        ]
        
        for pattern in medication_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.groups():
                    # Handle different pattern group structures
                    if pattern == r"(\bprescribe\s+([A-Z][a-z]+(?:an|in|ol|ide|ine|one|ate|zole|ium|oxin|arin|icin|mycin|dryl))\b)":
                        # For "prescribe X" pattern, use group 2 for medication name
                        if len(match.groups()) >= 2:
                            start = match.start(2)
                            end = match.end(2)
                            entity_text = match.group(2)
                        else:
                            continue
                    elif pattern == r"(\b(sumatriptan|ondansetron|ibuprofen|acetaminophen|metformin|lisinopril|atorvastatin|simvastatin|amlodipine|metoprolol|aspirin|tylenol)\b)":
                        # For common medications pattern, use group 2 for medication name
                        if len(match.groups()) >= 2:
                            start = match.start(2)
                            end = match.end(2)
                            entity_text = match.group(2)
                        else:
                            start = match.start(1)
                            end = match.end(1)
                            entity_text = match.group(1)
                    else:
                        # For other patterns, use group 1
                        start = match.start(1)
                        end = match.end(1)
                        entity_text = match.group(1)
                else:
                    start = match.start()
                    end = match.end()
                    entity_text = match.group()
                
                # Skip common instruction words that aren't medications
                instruction_words = ['it', 'them', 'this', 'that', 'these', 'those', 'they']
                if entity_text.lower() in instruction_words:
                    continue
                
                # Verify this is a real medication using our dictionary or RxNorm
                if hasattr(self, 'medication_dict') and entity_text.lower() in self.medication_dict:
                    if not self.medication_dict[entity_text.lower()]:
                        continue  # Skip if it's explicitly not a medication
                
                entities.append({
                    'text': entity_text,
                    'start': start,
                    'end': end,
                    'entity_type': 'MEDICATION',
                    'confidence': 0.7,
                    'method': 'rule-based'
                })
        
        # Apply symptom patterns
        symptom_patterns = [
            r"(\b(?:pain|ache|discomfort)\s+in\s+(?:my|the)\s+(\w+))",
            r"(\b(?:feeling|feel|felt)\s+(dizzy|nauseous|tired|weak|sick|lightheaded|confused|anxious|depressed))", # More specific symptom words
            r"(\b(headache|migraine|nausea|vomiting|dizziness|fatigue|fever|cough)s?\b)"
        ]
        
        for pattern in symptom_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.groups():
                    # If pattern has groups, use appropriate group
                    if pattern == r"(\b(?:pain|ache|discomfort)\s+in\s+(?:my|the)\s+(\w+))":
                        start = match.start()
                        end = match.end()
                        entity_text = match.group()
                    else:
                        start = match.start(1)
                        end = match.end(1)
                        entity_text = match.group(1)
                else:
                    start = match.start()
                    end = match.end()
                    entity_text = match.group()
                
                entities.append({
                    'text': entity_text,
                    'start': start,
                    'end': end,
                    'entity_type': 'SYMPTOM',
                    'confidence': 0.8,
                    'method': 'rule-based'
                })
        
        # Apply medical condition patterns
        condition_patterns = [
            r"\b(migraine|hypertension|diabetes|asthma|arthritis|depression|anxiety)\b",
            r"\b(diagnosed with|suffering from|have)\s+(\w+\s*\w*)\b"
        ]
        
        for pattern in condition_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if pattern == r"\b(diagnosed with|suffering from|have)\s+(\w+\s*\w*)\b" and len(match.groups()) > 1:
                    start = match.start(2)
                    end = match.end(2)
                    entity_text = match.group(2)
                else:
                    start = match.start(1) if len(match.groups()) > 0 else match.start()
                    end = match.end(1) if len(match.groups()) > 0 else match.end()
                    entity_text = match.group(1) if len(match.groups()) > 0 else match.group()
                
                entities.append({
                    'text': entity_text,
                    'start': start,
                    'end': end,
                    'entity_type': 'CONDITION',
                    'confidence': 0.8,
                    'method': 'rule-based'
                })
        
        return entities

    def _apply_dictionaries(self, text: str) -> List[Dict[str, Any]]:
        """Apply dictionary-based lookup to identify medical entities"""
        entities = []
        
        if hasattr(self.terminology_resolver, 'find_term_matches'):
            # Use terminology resolver to find matches
            try:
                dict_matches = self.terminology_resolver.find_term_matches(text)
                
                for match in dict_matches:
                    entity = {
                        'text': match['term'],
                        'start': match['start'],
                        'end': match['end'],
                        'entity_type': match['category'],
                        'confidence': 0.9,
                        'method': 'dictionary'
                    }
                    
                    # Add code if available
                    if 'umls_id' in match:
                        entity['code'] = match['umls_id']
                        entity['code_system'] = 'UMLS'
                        
                    entities.append(entity)
            except Exception as e:
                print(f"Error in terminology lookup: {e}")
        
        return entities

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities based on overlap and confidence"""
        if not entities:
            return []
            
        # Sort entities by confidence (highest first)
        sorted_entities = sorted(entities, key=lambda e: e.get('confidence', 0), reverse=True)
        
        # Filter out overlapping entities with lower confidence
        filtered_entities = []
        for entity in sorted_entities:
            should_add = True
            entity_range = range(entity['start'], entity['end'])
            
            # Check if this entity overlaps with any already added entity
            for added_entity in filtered_entities:
                added_range = range(added_entity['start'], added_entity['end'])
                
                # Check for overlap
                if (entity['start'] in added_range or
                    entity['end']-1 in added_range or
                    added_entity['start'] in entity_range or
                    added_entity['end']-1 in entity_range):
                    
                    # If there's overlap, check if this is an exact match
                    if (entity['start'] == added_entity['start'] and
                        entity['end'] == added_entity['end'] and
                        entity['entity_type'] == added_entity['entity_type']):
                        should_add = False
                        break
                        
                    # If there's overlap but not exact match, only keep it if confidence is higher
                    # and we want to keep the lower confidence one
                    if entity.get('confidence', 0) <= added_entity.get('confidence', 0):
                        should_add = False
                        break
            
            if should_add:
                filtered_entities.append(entity)
        
        return filtered_entities

    def _map_entity_type(self, spacy_type: str) -> str:
        """Map spaCy entity types to our medical entity types"""
        # Map common spaCy entity types to our categories
        mapping = {
            'DRUG': 'MEDICATION',
            'MEDICINE': 'MEDICATION',
            'CHEMICAL': 'MEDICATION',
            'DISEASE': 'CONDITION',
            'ILLNESS': 'CONDITION',
            'SYMPTOM': 'SYMPTOM',
            'BODY_PART': 'ANATOMY',
            'DATE': 'TEMPORAL',
            'TIME': 'TEMPORAL',
            'CARDINAL': 'DOSAGE',
            'QUANTITY': 'DOSAGE',
            'PERCENT': 'DOSAGE',
            'PROCEDURE': 'PROCEDURE',
            'TREATMENT': 'TREATMENT',
            'TEST': 'TEST'
        }
        
        return mapping.get(spacy_type, None)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical entities from text
        
        Args:
            text: Text to process
            
        Returns:
            List of entities
        """
        if isinstance(text, dict):
            # Extract text from structured transcript
            full_text = ""
            if "speaker_turns" in text:
                for turn in text["speaker_turns"]:
                    if "text" in turn:
                        full_text += turn["text"] + " "
            text = full_text.strip()
        
        entities = []
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Get entities from spaCy NER
        for ent in doc.ents:
            # Map spaCy entity types to our medical entity types
            entity_type = self._map_entity_type(ent.label_)
            if entity_type:
                entities.append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'entity_type': entity_type,
                    'confidence': 0.7,
                    'method': 'spacy'
                })
        
        # Get entities from rule-based patterns
        rule_entities = self._apply_rules(text)
        entities.extend(rule_entities)
        
        # Get entities from medical dictionaries
        dict_entities = self._apply_dictionaries(text)
        entities.extend(dict_entities)
        
        # Deduplicate entities
        entities = self._deduplicate_entities(entities)
        
        # Filter out non-medications that were incorrectly classified
        entities = self._filter_entities(entities, text)
        
        # Use contextual analysis to improve entity recognition
        entities = self._apply_contextual_analysis(entities, text)
        
        return entities

    def _filter_entities(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """
        Filter out incorrectly classified entities
        
        Args:
            entities: List of extracted entities
            text: Original text
            
        Returns:
            Filtered list of entities
        """
        filtered_entities = []
        
        # Common false-positive phrases that should never be entities
        stopwords = ["you taken", "got it", "have you", "i tried", "my mom", "could this", 
                    "how's your", "let's", "that's", "i'll", "we'll", "i'm", "it's"]
        
        # Question phrases that should be excluded
        question_patterns = [
            r"have you",
            r"did you",
            r"are you",
            r"could you",
            r"would you",
            r"do you",
            r"how (often|much|many|severe|long)",
            r"what (if|about|kind|type|medication)"
        ]
        
        # Compile patterns for faster matching
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in question_patterns]
        
        for entity in entities:
            entity_text = entity['text'].lower()
            entity_type = entity['entity_type']
            
            # 1. Skip entities that match common false-positive phrases
            if any(stopword in entity_text.lower() for stopword in stopwords):
                continue
                
            # 2. Skip entities that are part of question patterns
            if any(pattern.search(entity_text) for pattern in compiled_patterns):
                continue
            
            # 3. Check medication entities against our dictionary
            if entity_type == 'MEDICATION':
                # Just include all medications for now
                filtered_entities.append(entity)
                
            # 4. Verify entity based on type-specific checks
            elif entity_type == 'CONDITION':
                # Skip if it's a common non-condition word
                non_conditions = ["taken", "this", "that", "it", "there", "here", "they", "them"]
                if entity_text in non_conditions:
                    continue
                    
                # REMOVED PROBLEMATIC CALLS TO _query_umls and _query_snomed
                # Instead, rely on context for condition verification
                
                # Check surrounding context for condition indicators
                context_start = max(0, entity['start'] - 40)
                context_end = min(len(text), entity['end'] + 40)
                context = text[context_start:context_end].lower()
                
                condition_indicators = ["diagnosed with", "suffering from", "condition", 
                                    "disease", "disorder", "syndrome", "chronic"]
                
                if any(indicator in context for indicator in condition_indicators):
                    filtered_entities.append(entity)
                    continue
                    
                # Skip if the context suggests it's not a real condition
                if any(phrase in context for phrase in ["have you", "did you", "medication for"]):
                    continue
                    
            # Include all other properly classified entities
            else:
                filtered_entities.append(entity)
            
        # Additional refinement: merge related entities
        merged_entities = self._merge_related_entities(filtered_entities, text)  # Pass the text parameter
        
        return merged_entities

    

    def _merge_related_entities(self, entities, text):  # Added text parameter
        """
        Merge related entities like 'migraines' and 'head injuries' into more coherent entities
        
        Args:
            entities: List of entities to merge
            text: Original text for context
            
        Returns:
            List of merged entities
        """
        merged = []
        skip_indices = set()
        
        for i, entity in enumerate(entities):
            if i in skip_indices:
                continue
                
            # Look for entities that should be merged
            for j, other in enumerate(entities):
                if i == j or j in skip_indices:
                    continue
                    
                # Check if they're adjacent or very close
                if abs(entity['end'] - other['start']) <= 3 or abs(entity['start'] - other['end']) <= 3:
                    # Check if they're related by conjunctions (and, or)
                    start_pos = min(entity['start'], other['start'])
                    end_pos = max(entity['end'], other['end'])
                    
                    # If they're the same entity type and close, consider merging
                    if entity['entity_type'] == other['entity_type']:
                        span_text = text[start_pos:end_pos]
                        
                        # Check for conjunctions
                        if " or " in span_text or " and " in span_text or "," in span_text:
                            # Create merged entity
                            merged_entity = {
                                'text': span_text,
                                'start': start_pos,
                                'end': end_pos,
                                'entity_type': entity['entity_type'],
                                'confidence': max(entity.get('confidence', 0.5), other.get('confidence', 0.5)),
                                'method': 'merged'
                            }
                            merged.append(merged_entity)
                            skip_indices.add(j)
                            break
            
            # If no merge happened, keep original entity
            if i not in skip_indices:
                merged.append(entity)
        
        return merged
    
    def _apply_contextual_analysis(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """
        Apply contextual analysis to improve entity recognition
        
        Args:
            entities: List of extracted entities
            text: Original text
            
        Returns:
            Enhanced list of entities with context
        """
        for entity in entities:
            # Get the surrounding context (50 chars before and after)
            start = max(0, entity['start'] - 50)
            end = min(len(text), entity['end'] + 50)
            context = text[start:end]
            
            # Add context to the entity
            entity['context'] = context.strip()
            
            # Find modifiers in context
            if entity['entity_type'] == 'MEDICATION':
                modifiers = self._extract_medication_modifiers(context, entity['text'])
                if modifiers:
                    entity['modifiers'] = modifiers
        
        return entities
    
    def _extract_medication_modifiers(self, context: str, medication: str) -> Dict[str, Any]:
        """Extract medication modifiers from context"""
        modifiers = {}
        
        # Check for negation
        negation_terms = ["stop", "avoid", "reduce", "don't take", "do not take", 
                         "discontinue", "no more", "not", "without"]
        for term in negation_terms:
            if term.lower() in context.lower():
                modifiers['negation'] = True
                modifiers['negation_term'] = term
                break
                
        # Check for dosage
        dosage_pattern = r"(\d+)\s?(mg|milligram|g|gram|tablet|pill|capsule)"
        dosage_matches = re.findall(dosage_pattern, context, re.IGNORECASE)
        if dosage_matches:
            modifiers['dosage'] = dosage_matches[0][0] + " " + dosage_matches[0][1]
            
        # Check for frequency
        frequency_terms = ["daily", "once a day", "twice a day", "three times a day", 
                         "every day", "every hour", "every 4 hours", "every 6 hours", 
                         "weekly", "monthly", "as needed", "before meals", "after meals"]
        for term in frequency_terms:
            if term.lower() in context.lower():
                modifiers['frequency'] = term
                break
                
        return modifiers

    def normalize_and_map_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize and map entities to standard terminology
        
        Args:
            entities: List of extracted entities
            
        Returns:
            List of normalized and mapped entities
        """
        from .hybrid_entity_mapper import HybridEntityMapper
        
        mapper = HybridEntityMapper()
        return mapper.map_entities(entities)

    def _initialize_patterns(self):
        """Initialize patterns for medical entity recognition"""
        # Define entity types and common patterns
        self.compiled_patterns = {}
        
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
        for pattern_type, patterns in [
            ("MEDICATION", self.medication_patterns),
            ("SYMPTOM", self.symptom_patterns),
            ("CONDITION", self.condition_patterns),
            ("TEMPORAL", self.temporal_patterns)
        ]:
            self.compiled_patterns[pattern_type] = [re.compile(p, re.IGNORECASE) for p in patterns]

    