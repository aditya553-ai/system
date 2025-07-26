# model_manager.py
import os
import sys
import logging
import importlib
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
import spacy

class ModelManager:
    """Singleton class to manage model loading and sharing"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.terminologies = {}
            cls._instance.initialized = False
        return cls._instance
    
    def initialize_models(self, use_fast_resolver=False):
        """Initialize all models"""
        if self.initialized:
            return
            
        logging.info("Initializing shared models...")
        
        # 1. Load spaCy model once
        self._initialize_spacy()
        
        # 2. Load sentence transformer model if using fast resolver
        if use_fast_resolver:
            self._initialize_sentence_transformer()
        
        # 3. Load terminology dictionaries
        self._initialize_terminology_dictionaries()
          # 4. Initialize fast resolver if needed
        if use_fast_resolver:
            self._initialize_fast_resolver()
        
        # 5. Initialize relation extraction model (TEMPORARILY DISABLED - CAUSES HANG)
        # self._initialize_relation_model()
        logging.info("Skipping relation model initialization to avoid hang")
        
        self.initialized = True
        logging.info("Model manager initialization complete")

    def create_pipeline(self, use_fast_resolver=False):
        """Create and initialize a complete MedicalNLPPipeline"""
        # Make sure models are initialized
        if not self.initialized:
            self.initialize_models(use_fast_resolver=use_fast_resolver)
        
        # Import pipeline class
        import os
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        # Import and create pipeline
        try:
            from pipeline import MedicalNLPPipeline
            
            # Create a new instance with initialization
            pipeline = MedicalNLPPipeline(use_fast_resolver=use_fast_resolver)
            
            # Ensure entity recognizer is set
            if not hasattr(pipeline, 'entity_recognizer') or pipeline.entity_recognizer is None:
                logging.warning("Entity recognizer not initialized, using fallback")
                pipeline.entity_recognizer = self._create_fallback_entity_recognizer()
            
            logging.info("Created complete MedicalNLPPipeline")
            return pipeline
        except Exception as e:
            import traceback
            logging.error(f"Error creating pipeline: {str(e)}")
            traceback.print_exc()
            # Return a minimal pipeline that can still process transcripts
            return self._create_minimal_pipeline()

    def _create_fallback_entity_recognizer(self):
        """Create a minimal entity recognizer when the real one fails"""
        class MinimalEntityRecognizer:
            def extract_entities(self, text):
                """Extract a minimal set of entities from text"""
                import re
                entities = []
                
                # Simple pattern matching for basic entity types
                patterns = {
                    'MEDICATION': [r'\b(aspirin|tylenol|ibuprofen|advil|paracetamol)\b'],
                    'SYMPTOM': [r'\b(pain|headache|nausea|fever|cough|fatigue)\b'],
                    'CONDITION': [r'\b(hypertension|diabetes|asthma|migraine|arthritis)\b']
                }
                
                # Apply patterns
                for entity_type, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        for match in re.finditer(pattern, text, re.IGNORECASE):
                            entities.append({
                                'text': match.group(1),
                                'start': match.start(1),
                                'end': match.end(1),
                                'entity_type': entity_type,
                                'confidence': 0.7,
                                'method': 'fallback'
                            })
                
                return entities
        
        return MinimalEntityRecognizer()

    def _create_minimal_pipeline(self):
        """Create a minimal pipeline when the full pipeline cannot be initialized"""
        import json
        from datetime import datetime
        import os
        
        class MinimalPipeline:
            def process_transcript(self, transcript_data, output_dir=None):
                """Minimal implementation of transcript processing"""
                # Ensure output directory
                if output_dir is None:
                    import tempfile
                    output_dir = tempfile.mkdtemp()
                os.makedirs(output_dir, exist_ok=True)
                
                # Create a fallback note
                text = transcript_data
                if isinstance(text, dict):
                    text = json.dumps(text)
                elif not isinstance(text, str):
                    text = str(text)
                
                # Create a basic SOAP note
                basic_note = {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "subjective": {
                        "chief_complaint": "Generated from transcript (fallback mode)",
                        "history_of_present_illness": text[:500] if len(text) > 500 else text
                    },
                    "objective": {
                        "vitals": "Not assessed in fallback mode",
                        "physical_exam": "Not assessed in fallback mode"
                    },
                    "assessment": {
                        "diagnosis": {
                            "primary_diagnosis": "Unable to determine (fallback mode)",
                            "differential_diagnoses": "Not assessed in fallback mode"
                        }
                    },
                    "plan": {
                        "plan_text": "Please review transcript for proper assessment"
                    }
                }
                
                # Save note to file
                note_path = os.path.join(output_dir, "fallback_clinical_note.json")
                with open(note_path, 'w', encoding='utf-8') as f:
                    json.dump(basic_note, f, indent=2)
                
                # Return result
                return {
                    "notes": {
                        "json": note_path
                    }
                }
        
        logging.warning("Created minimal fallback pipeline")
        return MinimalPipeline()
    
    def _initialize_spacy(self):
        """Initialize spaCy model"""
        try:
            self.models["spacy"] = spacy.load("en_core_web_sm")
            logging.info("Loaded spaCy model")
        except Exception as e:
            logging.error(f"Error loading spaCy model: {e}")
            try:
                # Try to load as a direct import
                import en_core_web_sm
                self.models["spacy"] = en_core_web_sm.load()
                logging.info("Loaded spaCy model from direct import")
            except Exception as inner_e:
                logging.error(f"Failed to load spaCy model via direct import: {inner_e}")
                self.models["spacy"] = None
    
    def _initialize_sentence_transformer(self):
        """Initialize sentence transformer model"""
        try:
            logging.info("Loading shared SentenceTransformer model")
            model_name = "pritamdeka/S-BioBert-snli-multinli-stsb"
            self.models["sentence_transformer"] = SentenceTransformer(model_name)
            logging.info(f"Loaded shared SentenceTransformer model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading SentenceTransformer: {e}")
            self.models["sentence_transformer"] = None
    
    def _initialize_fast_resolver(self):
        """Initialize fast terminology resolver"""
        if not self.models["sentence_transformer"]:
            logging.warning("Cannot initialize fast resolver without sentence transformer")
            return
            
        try:
            # Add the current directory to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.append(current_dir)
            
            # Try to import FastTerminologyResolver
            logging.info("Initializing fast resolver...")
            
            # Try alternative import paths
            try:
                from terminology.fast_resolver import FastTerminologyResolver
            except ImportError:
                try:
                    from fast_resolver import FastTerminologyResolver
                except ImportError:
                    # Try to import from full path
                    sys.path.append(os.path.join(current_dir, "terminology"))
                    from fast_resolver import FastTerminologyResolver
            
            # Initialize the fast resolver
            self.models["fast_resolver"] = FastTerminologyResolver(
                sentence_transformer=self.models["sentence_transformer"]
            )
            
            # Index the terminologies
            if self.models["fast_resolver"]:
                self.models["fast_resolver"].build_indices(self.terminologies)
                logging.info("Fast resolver initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing fast resolver: {e}")
            self.models["fast_resolver"] = None
    
    def _initialize_relation_model(self):
        """Initialize relation extraction model"""
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            logging.info("Loading relation extraction model...")
            
            model_name = "bvanaken/clinical-assertion-negation-bert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            self.models["relation_tokenizer"] = tokenizer
            self.models["relation_model"] = model
            
            logging.info("Relation extraction model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading relation extraction model: {e}")
            self.models["relation_tokenizer"] = None
            self.models["relation_model"] = None
    
    def _initialize_terminology_dictionaries(self):
        """Load terminology dictionaries"""
        import os
        import csv
        
        # Initialize empty dictionaries
        self.terminologies = {
            "medications": [],
            "conditions": [],
            "lab_tests": [],
            "symptoms": []
        }
        
        try:
            # Get path to terminology dictionaries
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dictionary_dir = os.path.join(current_dir, "terminology", "dictionaries")
            
            # Load each dictionary
            for term_type in self.terminologies.keys():
                dict_path = os.path.join(dictionary_dir, f"{term_type}.csv")
                
                if os.path.exists(dict_path):
                    try:
                        with open(dict_path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                self.terminologies[term_type].append(row)
                        logging.info(f"Loaded {len(self.terminologies[term_type])} terms from {dict_path}")
                    except Exception as e:
                        logging.error(f"Error reading {term_type} dictionary: {e}")
                else:
                    logging.warning(f"Dictionary file not found: {dict_path}")
                    # Add fallback entries if the file doesn't exist
                    self._add_fallback_terms(term_type)
        except Exception as e:
            logging.error(f"Error loading terminology dictionaries: {e}")
            # Add fallback entries
            self._add_fallback_terms()
    
    def _add_fallback_terms(self, specific_type=None):
        """Add fallback terminology entries"""
        # Basic fallback dictionaries
        fallbacks = {
            "medications": ["ibuprofen", "acetaminophen", "lisinopril", "metformin", "atorvastatin"],
            "conditions": ["hypertension", "diabetes", "migraine", "asthma"],
            "lab_tests": ["CBC", "CMP", "A1C", "lipid panel"],
            "symptoms": ["headache", "fever", "cough", "nausea", "fatigue"]
        }
        
        # Add entries for specific type or all if not specified
        types_to_add = [specific_type] if specific_type else fallbacks.keys()
        
        for term_type in types_to_add:
            if term_type in fallbacks:
                for term in fallbacks[term_type]:
                    self.terminologies[term_type].append({
                        "term": term, 
                        "code": "", 
                        "synonyms": ""
                    })
                
                if len(self.terminologies[term_type]) > 0:
                    logging.warning(f"Added {len(fallbacks[term_type])} fallback terms for {term_type}")
    
    def get_model(self, model_name):
        """Get a loaded model by name"""
        if not self.initialized:
            logging.warning("Model manager not initialized; initializing with default settings")
            self.initialize_models()
            
        return self.models.get(model_name)
    
    def get_terminology(self, terminology_name):
        """Get a loaded terminology dictionary"""
        if not self.initialized:
            logging.warning("Model manager not initialized; initializing with default settings")
            self.initialize_models()
            
        return self.terminologies.get(terminology_name)
    
    def get_fast_resolver(self):
        """Get the fast terminology resolver if available"""
        return self.models.get("fast_resolver")