import os
import csv
import json
import re
import logging
from typing import Dict, List, Any, Optional
from fuzzywuzzy import process, fuzz
from model_manager import ModelManager

# Import fast resolver if available
FAST_RESOLVER_AVAILABLE = False
try:
    from .fast_terminology_resolver import FastTerminologyResolver
    FAST_RESOLVER_AVAILABLE = True
    print("Fast terminology resolver is available")
except ImportError:
    print("Fast terminology resolver not available. Will use dictionary-based methods.")

class TerminologyResolver:
    """
    Class for resolving medical terminology to standard codes and ontologies.
    """
    
    def __init__(self, model_manager_instance: ModelManager, use_cache: bool = True, use_fast_resolver: bool = True):
        """
        Initialize the terminology resolver with various medical dictionaries
        and APIs for term resolution.
        
        Args:
            use_cache: Whether to use caching for resolved terms
            use_fast_resolver: Whether to use fast vector-based resolver
        """
        self.logger = logging.getLogger(__name__)
        self.use_cache = use_cache
        self.cache = {}
        
        # Set up dictionaries
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.dictionaries = {}
        # Get or create model manager

        self.model_manager = model_manager_instance
        
        # Use the model manager to access terminologies and models
        self.medications = self.model_manager.get_terminology("medications") or []
        self.conditions = self.model_manager.get_terminology("conditions") or []
        self.lab_tests = self.model_manager.get_terminology("lab_tests") or []
        self.symptoms = self.model_manager.get_terminology("symptoms") or []
        
        # Use fast resolver if available
        self.use_fast_resolver = use_fast_resolver
        self.fast_resolver = self.model_manager.get_fast_resolver() if use_fast_resolver else None

    def _query_umls(self, term):
        """Query UMLS database for a term"""
        # Simple implementation to get things working
        # This just checks if the term is in our known medical terms
        term = term.lower().strip()
        for category in self.terminology_dict:
            for entry in self.terminology_dict[category]:
                if term == entry["term"].lower() or term in entry.get("synonyms", []):
                    return True
        return False

    def _query_snomed(self, term):
        """Query SNOMED database for a term"""
        # Simple implementation for now
        # Just delegates to _query_umls as a fallback
        return self._query_umls(term)
    
    def _load_specialty_dictionaries(self):
        """Load specialty dictionaries for various entity types"""
        dict_path = os.path.join(self.script_dir, "dictionaries")
        os.makedirs(dict_path, exist_ok=True)
        
        # Create sample dictionaries if needed
        self._create_sample_dictionaries()
        
        # Load dictionaries from CSV files
        dict_files = {
            "medications": os.path.join(dict_path, "medications.csv"),
            "conditions": os.path.join(dict_path, "conditions.csv"),
            "lab_tests": os.path.join(dict_path, "lab_tests.csv"),
            "symptoms": os.path.join(dict_path, "symptoms.csv")
        }
        
        for key, file_path in dict_files.items():
            self.dictionaries[key] = {}
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            term = row.get('term', '').lower()
                            if term:
                                self.dictionaries[key][term] = row
                    print(f"Loaded {len(self.dictionaries[key])} terms from {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    def _create_sample_dictionaries(self):
        """Create sample dictionaries if they don't exist"""
        dict_path = os.path.join(self.script_dir, "dictionaries")
        
        # Medications sample
        med_file = os.path.join(dict_path, "medications.csv")
        if not os.path.exists(med_file):
            with open(med_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['term', 'preferred_term', 'rxnorm_code', 'atc_code', 'class'])
                writer.writerow(['ibuprofen', 'Ibuprofen', '5640', 'M01AE01', 'NSAID'])
                writer.writerow(['advil', 'Ibuprofen', '5640', 'M01AE01', 'NSAID'])
                writer.writerow(['motrin', 'Ibuprofen', '5640', 'M01AE01', 'NSAID'])
                writer.writerow(['ibuproven', 'Ibuprofen', '5640', 'M01AE01', 'NSAID'])
                writer.writerow(['aspirin', 'Aspirin', '1191', 'N02BA01', 'NSAID'])
                writer.writerow(['acetaminophen', 'Acetaminophen', '161', 'N02BE01', 'Non-NSAID analgesic'])
                writer.writerow(['tylenol', 'Acetaminophen', '161', 'N02BE01', 'Non-NSAID analgesic'])
            print(f"Created sample medications dictionary at {med_file}")
        
        # Conditions sample
        cond_file = os.path.join(dict_path, "conditions.csv")
        if not os.path.exists(cond_file):
            with open(cond_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['term', 'preferred_term', 'icd10_code', 'snomed_code', 'system'])
                writer.writerow(['migraine', 'Migraine', 'G43.909', '37796009', 'nervous'])
                writer.writerow(['migraines', 'Migraine', 'G43.909', '37796009', 'nervous'])
                writer.writerow(['headache', 'Headache', 'R51', '25064002', 'nervous'])
                writer.writerow(['headaches', 'Headache', 'R51', '25064002', 'nervous'])
            print(f"Created sample conditions dictionary at {cond_file}")
        
        # Lab tests sample
        lab_file = os.path.join(dict_path, "lab_tests.csv")
        if not os.path.exists(lab_file):
            with open(lab_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['term', 'preferred_term', 'loinc_code', 'component', 'normal_range'])
                writer.writerow(['glucose', 'Glucose measurement', '2345-7', 'Glucose', '70-100 mg/dL'])
                writer.writerow(['blood sugar', 'Glucose measurement', '2345-7', 'Glucose', '70-100 mg/dL'])
                writer.writerow(['cbc', 'Complete blood count', '58410-2', 'Blood count', ''])
                writer.writerow(['complete blood count', 'Complete blood count', '58410-2', 'Blood count', ''])
            print(f"Created sample lab tests dictionary at {lab_file}")
        
        # Symptoms sample
        symptom_file = os.path.join(dict_path, "symptoms.csv")
        if not os.path.exists(symptom_file):
            with open(symptom_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['term', 'preferred_term', 'icd10_code', 'snomed_code', 'system'])
                writer.writerow(['pain', 'Pain', 'R52', '22253000', 'general'])
                writer.writerow(['nausea', 'Nausea', 'R11.0', '422587007', 'digestive'])
                writer.writerow(['dizziness', 'Dizziness', 'R42', '404640003', 'nervous'])
                writer.writerow(['fatigue', 'Fatigue', 'R53.83', '84229001', 'general'])
                writer.writerow(['sensitivity to light', 'Photophobia', 'H53.14', '13791008', 'nervous'])
            print(f"Created sample symptoms dictionary at {symptom_file}")
    
    def resolve_entity(self, text: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """
        Resolve an entity to its standard terminology using the appropriate method.
        
        Args:
            text: The entity text to resolve
            entity_type: The type of entity
            
        Returns:
            Dict with resolved entity information or None if not resolved
        """
        if not text:
            return None
            
        # Normalize entity type
        entity_type = entity_type.upper() if entity_type else ""
        
        # Check cache first if enabled
        cache_key = f"{text.lower()}:{entity_type}"
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try fast resolver first if available
        if self.use_fast_resolver and self.fast_resolver:
            try:
                result = self.fast_resolver.resolve_term(text, entity_type)
                if result and result.get('codes'):
                    # Format result for compatibility
                    formatted_result = {
                        'text': text,
                        'preferred_term': result.get('normalized', text),
                        'entity_type': entity_type,
                        'source': 'fast_resolver'
                    }
                    
                    # Add codes based on entity type
                    if entity_type in ["MEDICATION", "DRUG"] and 'rxnorm' in result['codes']:
                        formatted_result['rxnorm_code'] = result['codes']['rxnorm']
                    
                    if entity_type in ["CONDITION", "DIAGNOSIS", "SYMPTOM"]:
                        if 'icd10' in result['codes']:
                            formatted_result['icd10_code'] = result['codes']['icd10']
                        if 'snomed' in result['codes']:
                            formatted_result['snomed_code'] = result['codes']['snomed']
                    
                    if entity_type in ["LAB_TEST", "TEST"] and 'loinc' in result['codes']:
                        formatted_result['loinc_code'] = result['codes']['loinc']
                    
                    # Cache result if enabled
                    if self.use_cache:
                        self.cache[cache_key] = formatted_result
                    
                    return formatted_result
            except Exception as e:
                print(f"Fast resolver error for {text}: {e}")
        
        # Fall back to dictionary-based resolution
        result = self._dictionary_resolve(text, entity_type)
        
        # Cache result if enabled
        if self.use_cache and result:
            self.cache[cache_key] = result
            
        return result
    
    def _dictionary_resolve(self, text: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """
        Resolve an entity using dictionary lookup
        
        Args:
            text: The entity text to resolve
            entity_type: The type of entity
            
        Returns:
            Dict with resolved entity information or None
        """
        # Normalize text
        text_lower = text.lower()
        
        # Map entity type to dictionary
        dict_key = self._map_entity_type_to_dict_key(entity_type)
        if not dict_key or dict_key not in self.dictionaries:
            return self._create_minimal_result(text, entity_type)
        
        # Look for exact match first
        if text_lower in self.dictionaries[dict_key]:
            return self._format_dictionary_result(text, self.dictionaries[dict_key][text_lower], entity_type)
        
        # Try fuzzy matching
        matches = process.extract(text_lower, self.dictionaries[dict_key].keys(), limit=3)
        for term, score in matches:
            if score >= 85:  # Good match threshold
                return self._format_dictionary_result(text, self.dictionaries[dict_key][term], entity_type)
        
        # Return minimal info if no match found
        return self._create_minimal_result(text, entity_type)
    
    def _map_entity_type_to_dict_key(self, entity_type: str) -> Optional[str]:
        """Map entity type to dictionary key"""
        entity_type = entity_type.upper() if entity_type else ""
        
        mapping = {
            "MEDICATION": "medications",
            "DRUG": "medications",
            "TREATMENT": "medications",
            "CONDITION": "conditions",
            "DISEASE": "conditions",
            "DIAGNOSIS": "conditions",
            "SYMPTOM": "symptoms",
            "LAB_TEST": "lab_tests",
            "TEST": "lab_tests"
        }
        
        return mapping.get(entity_type)
    
    def _format_dictionary_result(self, original_text: str, dict_entry: Dict, entity_type: str) -> Dict[str, Any]:
        """Format dictionary entry as resolver result"""
        result = {
            'text': original_text,
            'preferred_term': dict_entry.get('preferred_term', original_text),
            'entity_type': entity_type,
            'source': 'dictionary'
        }
        
        # Add codes based on entity type
        if entity_type in ["MEDICATION", "DRUG"] and 'rxnorm_code' in dict_entry:
            result['rxnorm_code'] = dict_entry['rxnorm_code']
        
        if entity_type in ["CONDITION", "DIAGNOSIS", "SYMPTOM"]:
            if 'icd10_code' in dict_entry:
                result['icd10_code'] = dict_entry['icd10_code']
            if 'snomed_code' in dict_entry:
                result['snomed_code'] = dict_entry['snomed_code']
        
        if entity_type in ["LAB_TEST", "TEST"] and 'loinc_code' in dict_entry:
            result['loinc_code'] = dict_entry['loinc_code']
        
        return result
    
    def _create_minimal_result(self, text: str, entity_type: str) -> Dict[str, Any]:
        """Create minimal result when no match is found"""
        return {
            'text': text,
            'preferred_term': text,
            'entity_type': entity_type,
            'source': 'unresolved'
        }