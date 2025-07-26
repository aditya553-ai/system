import json
import os
import re
from typing import Dict, List, Any, Optional

# Import our resolvers, but make them optional
SCISPACY_AVAILABLE = False
try:
    from scispacy_terminology_resolver import ScispacyTerminologyResolver
    SCISPACY_AVAILABLE = True
    print("sciSpaCy module found and ready")
except ImportError:
    print("sciSpaCy not available, will use dictionary-based resolution")

class TerminologyResolver:
    """
    Class for resolving medical entities to standard terminology codes.
    Can map to various coding systems including:
    - SNOMED CT
    - ICD-10
    - RxNorm
    - LOINC
    - CPT
    """
    
    def __init__(self, terminologies=None, use_scispacy=True):
        """
        Initialize the resolver with a set of terminologies.
        
        Args:
            terminologies: Dict of terminology names to their data files or None to load defaults
            use_scispacy: Whether to use sciSpaCy for terminology resolution if available
        """
        self.terminology_data = {}
        self.use_scispacy = use_scispacy and SCISPACY_AVAILABLE
        
        # Initialize sciSpaCy resolver if available and requested
        self.sci_resolver = None
        if self.use_scispacy:
            try:
                print("Initializing sciSpaCy terminology resolver...")
                self.sci_resolver = ScispacyTerminologyResolver()
                print("sciSpaCy resolver ready!")
            except Exception as e:
                print(f"Could not initialize sciSpaCy resolver: {e}")
                print("Falling back to dictionary-based resolution.")
                self.use_scispacy = False
        
        # Default terminologies to load if none specified
        if terminologies is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_dir = os.path.join(script_dir, "data", "terminology")
            
            terminologies = {
                'rxnorm': os.path.join(default_dir, 'rxnorm_mapping.json'),
                'icd10': os.path.join(default_dir, 'icd10_mapping.json'),
                'snomed': os.path.join(default_dir, 'snomed_mapping.json'),
                'loinc': os.path.join(default_dir, 'loinc_mapping.json')
            }
        
        # Always load the dictionary-based terminology for fallback
        for name, path in terminologies.items():
            try:
                self._load_terminology(name, path)
            except Exception as e:
                print(f"Failed to load terminology {name} from {path}: {e}")
                self.terminology_data[name] = {}  # Initialize with empty dict
    
    def _load_terminology(self, name, path):
        """Load terminology data from a file"""
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.terminology_data[name] = json.load(f)
                print(f"Loaded {name} terminology with {len(self.terminology_data[name])} entries")
            except Exception as e:
                print(f"Error loading {name} terminology from {path}: {e}")
                self.terminology_data[name] = {}
        else:
            print(f"Terminology file not found: {path}")
            self.terminology_data[name] = {}
    
    def resolve_entities(self, entities, entity_types=None):
        """
        Resolve a list of entities to standard terminology codes
        
        Args:
            entities: List of entity strings to resolve
            entity_types: Dict mapping entity text to entity type
            
        Returns:
            Dict mapping entity text to normalized form and codes
        """
        entity_types = entity_types or {}
        results = {}
        
        # Try sciSpaCy first if available
        if self.use_scispacy and self.sci_resolver:
            try:
                print("Using sciSpaCy for terminology resolution...")
                sci_results = self.sci_resolver.resolve_entities(entities, entity_types)
                
                # Check if we got valid results
                if sci_results:
                    # For any entities without codes, fall back to dictionary method
                    for entity in entities:
                        if entity in sci_results and sci_results[entity] and sci_results[entity].get("codes"):
                            results[entity] = sci_results[entity]
                        else:
                            # Fallback to dictionary for this entity
                            results[entity] = self._resolve_entity_with_dictionary(entity, entity_types.get(entity, ""))
                    
                    return results
            except Exception as e:
                print(f"Error using sciSpaCy resolver: {e}")
                print("Falling back to dictionary-based resolution for all entities.")
        
        # Dictionary-based resolution
        print("Using dictionary-based terminology resolution...")
        for entity in entities:
            entity_type = entity_types.get(entity, "")
            results[entity] = self._resolve_entity_with_dictionary(entity, entity_type)
        
        return results
    
    def _resolve_entity_with_dictionary(self, entity, entity_type):
        """
        Resolve a single entity using dictionary lookup
        
        Args:
            entity: Entity text to resolve
            entity_type: Type of entity
            
        Returns:
            Dictionary with normalized form and codes
        """
        # Set appropriate terminology systems based on entity type
        applicable_systems = self._get_applicable_systems(entity_type)
        
        result = {
            "original": entity,
            "normalized": entity,
            "codes": {},
            "type": entity_type
        }
        
        for system in applicable_systems:
            if system in self.terminology_data:
                # Try exact match
                if entity.lower() in self.terminology_data[system]:
                    result["codes"][system] = self.terminology_data[system][entity.lower()]
                else:
                    # Try partial matches
                    for term, code in self.terminology_data[system].items():
                        if entity.lower() in term or term in entity.lower():
                            result["codes"][system] = code
                            break
        
        return result
    
    def _get_applicable_systems(self, entity_type):
        """Get applicable terminology systems based on entity type"""
        entity_type = entity_type.lower() if entity_type else ""
        
        if entity_type in ["medication", "drug", "treatment"]:
            return ["rxnorm", "snomed"]
        elif entity_type in ["condition", "disease", "diagnosis"]:
            return ["icd10", "snomed"]
        elif entity_type in ["symptom", "problem"]:
            return ["snomed", "icd10"]
        elif entity_type in ["procedure", "test", "surgery"]:
            return ["snomed", "cpt", "loinc"]
        elif entity_type in ["lab", "laboratory", "test_result"]:
            return ["loinc"]
        else:
            # Default to checking all systems
            return ["snomed", "icd10", "rxnorm", "loinc"]