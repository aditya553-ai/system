import os
import sys
import json
import traceback
import time
import logging
from typing import Dict, List, Any

from sentence_transformers import SentenceTransformer
import spacy
from model_manager import ModelManager

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use absolute imports instead of relative imports
from entity_recognition.medical_entity_recognizer import MedicalEntityRecognizer
from entity_recognition.hybrid_entity_mapper import HybridEntityMapper
from entity_recognition.clinical_entity_recognizer import ClinicalEntityRecognizer
from relation_extraction.semantic_relation_extractor import SemanticRelationExtractor
from knowledge_graph.graph_builder import GraphBuilder
from knowledge_graph.graph_validator import GraphValidator
from llm.llm_client import LLMClient
from terminology.terminology_resolver import TerminologyResolver
from note_generation_bridge import NoteBridge
from error_correction.error_mitigation import TranscriptionErrorCorrector


class MedicalNLPPipeline:
    def __init__(self, use_fast_resolver=False):
        """Initialize pipeline components"""
        
        logging.info("Initializing MedicalNLPPipeline...")
        self.model_manager = ModelManager()
        self.model_manager.initialize_models(use_fast_resolver=use_fast_resolver)
        
        try:
            # Pass the created model_manager instance
            self.entity_recognizer = MedicalEntityRecognizer(model_manager_instance=self.model_manager)
            logging.info("Entity recognizer initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing entity recognizer: {str(e)}", exc_info=True) # Added exc_info=True for full traceback
            self.entity_recognizer = self._create_fallback_entity_recognizer()

        try:
            # ClinicalEntityRecognizer class is imported at the top of the file.
            # Ensure it's instantiated correctly. Pass model_manager if needed.
            self.clinical_recognizer = ClinicalEntityRecognizer(model_manager_instance=self.model_manager)
            logging.info("ClinicalEntityRecognizer initialized successfully.")
        except NameError as ne: 
            # This catches if ClinicalEntityRecognizer class itself is not defined (e.g., import failed)
            logging.error(f"ClinicalEntityRecognizer class not found or not imported. Clinical entity extraction will use fallback. Error: {ne}", exc_info=True)
            self.clinical_recognizer = None # IMPORTANT: Fallback to None
        except Exception as e:
            logging.error(f"Error initializing ClinicalEntityRecognizer: {str(e)}. Clinical entity extraction will use fallback.", exc_info=True)
            self.clinical_recognizer = None # IMPORTANT: Fallback to None if any other error occurs during init

        try:
            self.entity_mapper = HybridEntityMapper(model_manager_instance=self.model_manager)
            logging.info("HybridEntityMapper initialized successfully.")
        except NameError as ne: 
            # This catches if ClinicalEntityRecognizer class itself is not defined (e.g., import failed)
            logging.error(f"HybridEntityMapper class not found or not imported. Clinical entity extraction will use fallback. Error: {ne}", exc_info=True)
            self.entity_mapper = None # IMPORTANT: Fallback to None
        except Exception as e:
            logging.error(f"Error initializing HybridEntityMapper: {str(e)}. will use fallback.", exc_info=True)
            self.entity_mapper = None # IMPORTANT: Fallback to None if any other error occurs during init

        try:
            self.error_corrector = TranscriptionErrorCorrector(model_manager_instance=self.model_manager, use_llm=False)
            logging.info("TranscriptionErrorCorrector initialized successfully.")
        except NameError as ne: 
            # This catches if ClinicalEntityRecognizer class itself is not defined (e.g., import failed)
            logging.error(f"TranscriptionErrorCorrector class not found or not imported. will use fallback. Error: {ne}", exc_info=True)
            self.error_corrector = None # IMPORTANT: Fallback to None
        except Exception as e:
            logging.error(f"Error initializing TranscriptionErrorCorrector: {str(e)}. will use fallback.", exc_info=True)
            self.error_corrector = None # IMPORTANT: Fallback to None if any other error occurs during init

        try:
            self.relation_extractor = SemanticRelationExtractor(model_manager_instance=self.model_manager)
            logging.info("SemanticRelationExtractor initialized successfully.")
        except NameError as ne: 
            # This catches if ClinicalEntityRecognizer class itself is not defined (e.g., import failed)
            logging.error(f"SemanticRelationExtractor class not found or not imported. will use fallback. Error: {ne}", exc_info=True)
            self.relation_extractor = None # IMPORTANT: Fallback to None
        except Exception as e:
            logging.error(f"Error initializing SemanticRelationExtractor: {str(e)}. will use fallback.", exc_info=True)
            self.relation_extractor = None # IMPORTANT: Fallback to None if any other error occurs during init

        try:
            self.graph_builder = GraphBuilder(model_manager_instance=self.model_manager)
            logging.info("GraphBuilder initialized successfully.")
        except NameError as ne: 
            # This catches if ClinicalEntityRecognizer class itself is not defined (e.g., import failed)
            logging.error(f"GraphBuilder class not found or not imported. will use fallback. Error: {ne}", exc_info=True)
            self.graph_builder = None # IMPORTANT: Fallback to None
        except Exception as e:
            logging.error(f"Error initializing GraphBuilder: {str(e)}. will use fallback.", exc_info=True)
            self.graph_builder = None # IMPORTANT: Fallback to None if any other error occurs during init

        try:
            self.graph_validator = GraphValidator(model_manager_instance=self.model_manager)
            logging.info("GraphValidator initialized successfully.")
        except NameError as ne: 
            # This catches if ClinicalEntityRecognizer class itself is not defined (e.g., import failed)
            logging.error(f"GraphValidator class not found or not imported. will use fallback. Error: {ne}", exc_info=True)
            self.graph_validator = None # IMPORTANT: Fallback to None
        except Exception as e:
            logging.error(f"Error initializing GraphValidator: {str(e)}. will use fallback.", exc_info=True)
            self.graph_validator = None # IMPORTANT: Fallback to None if any other error occurs during init

        try:
            self.llm_client = LLMClient(model_manager_instance=self.model_manager)
            logging.info("LLMClient initialized successfully.")
        except NameError as ne: 
            # This catches if ClinicalEntityRecognizer class itself is not defined (e.g., import failed)
            logging.error(f"LLMClient class not found or not imported. will use fallback. Error: {ne}", exc_info=True)
            self.llm_client = None # IMPORTANT: Fallback to None
        except Exception as e:
            logging.error(f"Error initializing LLMClient: {str(e)}. will use fallback.", exc_info=True)
            self.llm_client = None # IMPORTANT: Fallback to None if any other error occurs during init

        try:
            self.terminology_resolver = TerminologyResolver(model_manager_instance=self.model_manager)
            logging.info("TerminologyResolver initialized successfully.")
        except NameError as ne: 
            # This catches if ClinicalEntityRecognizer class itself is not defined (e.g., import failed)
            logging.error(f"TerminologyResolver class not found or not imported. will use fallback. Error: {ne}", exc_info=True)
            self.terminology_resolver = None # IMPORTANT: Fallback to None
        except Exception as e:
            logging.error(f"Error initializing TerminologyResolver: {str(e)}. will use fallback.", exc_info=True)
            self.terminology_resolver = None # IMPORTANT: Fallback to None if any other error occurs during init
    def _convert_to_structured_entities(self, entities: list) -> dict:
        """
        Fallback method to convert flat entities to a basic structured format
        if ClinicalEntityRecognizer is not available.
        """
        logging.warning("ClinicalEntityRecognizer not available. Converting basic entities to a simple structured format.")
        structured_output = {
            "symptoms": [], "medications": [], "conditions": [], "other_findings": []
        }
        for entity in entities:
            text = entity.get("text")
            etype = entity.get("entity_type", "").upper()
            if etype == "SYMPTOM":
                structured_output["symptoms"].append({"term": text, "details": "Derived from general NER"})
            elif etype == "MEDICATION":
                structured_output["medications"].append({"name": text, "details": "Derived from general NER"})
            elif etype == "CONDITION":
                structured_output["conditions"].append({"term": text, "details": "Derived from general NER"})
            else:
                structured_output["other_findings"].append({"text": text, "type": etype, "details": "Derived from general NER"})
        return structured_output
            
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

    def initialize_models(self, use_fast_resolver=False):
        """Initialize all models"""
        if self.initialized:
            return
            
        logging.info("Initializing shared models...")
        
        # Load spaCy model once
        try:
            self.models["spacy"] = spacy.load("en_core_web_sm")
            logging.info("Loaded spaCy model")
        except Exception as e:
            logging.error(f"Error loading spaCy model: {e}")
            self.models["spacy"] = None
        
        # Load sentence transformer model once if using fast resolver
        if use_fast_resolver:
            try:
                logging.info("Loading shared SentenceTransformer model")
                model_name = "pritamdeka/S-BioBert-snli-multinli-stsb"
                self.models["sentence_transformer"] = SentenceTransformer(model_name)
                logging.info(f"Loaded shared SentenceTransformer model: {model_name}")
            except Exception as e:
                logging.error(f"Error loading SentenceTransformer: {e}")
                self.models["sentence_transformer"] = None
        
        # Load terminology dictionaries once
        try:
            self.load_terminology_dictionaries()
        except Exception as e:
            logging.error(f"Error loading terminology dictionaries: {e}")
            # Use fallback dictionaries
            self.ensure_terminology_dictionaries()
        
        self.initialized = True
        logging.info("Model manager initialization complete")

    def generate_clinical_notes(self, result, output_dir=None):
        """
        Generate clinical notes from the processing results
        
        Args:
            result: The result from the process_transcript method
            output_dir: Directory to save the notes (optional)
            
        Returns:
            Dictionary with paths to generated notes
        """
        print("\nStep 6: Generating clinical notes...")
        note_time = time.time()
        
        # Initialize note bridge
        note_bridge = NoteBridge()
        
        # Generate the note
        note_paths = note_bridge.generate_note(result, output_dir)
        
        print(f"Generated clinical notes in {time.time() - note_time:.2f} seconds")
        return note_paths
    def print_structured_entities(self, entities_dict):
        """Print entities in a structured format for clinical review"""
        print("\n===== STRUCTURED CLINICAL ENTITIES WITH CODES =====\n")
        
        # Helper function to extract all codes from an entity
        def get_code_string(entity):
            codes = []
            if 'icd10_code' in entity:
                codes.append(f"ICD-10: {entity['icd10_code']}")
            if 'snomed_code' in entity:
                codes.append(f"SNOMED: {entity['snomed_code']}")
            if 'rxnorm_code' in entity:
                codes.append(f"RxNorm: {entity['rxnorm_code']}")
            if 'loinc_code' in entity:
                codes.append(f"LOINC: {entity['loinc_code']}")
            if 'cpt_code' in entity:
                codes.append(f"CPT: {entity['cpt_code']}")
            if 'code' in entity and 'code_system' in entity:
                codes.append(f"{entity['code_system']}: {entity['code']}")
            
            # Add any other code types that might be present
            for key, value in entity.items():
                if key.endswith('_code') and key not in ['icd10_code', 'snomed_code', 'rxnorm_code', 'loinc_code', 'cpt_code']:
                    codes.append(f"{key.replace('_code', '').upper()}: {value}")
                    
            return f" ({', '.join(codes)})" if codes else ""
        
        # Print symptoms section
        print("SYMPTOMS:")
        if entities_dict["SYMPTOMS"]:
            for entity in entities_dict["SYMPTOMS"]:
                code_str = get_code_string(entity)
                print(f"- {entity.get('text', '')}{code_str} [{entity.get('confidence', 0.0):.1f}]")
        else:
            print("- None identified")
        
        # Print negated symptoms
        print("\nNEGATED SYMPTOMS:")
        if entities_dict["NEGATED_SYMPTOMS"]:
            for entity in entities_dict["NEGATED_SYMPTOMS"]:
                code_str = get_code_string(entity)
                print(f"- {entity.get('negated_term', entity.get('text', ''))}{code_str} [{entity.get('confidence', 0.0):.1f}]")
        else:
            print("- None identified")
        
        # Print medications
        print("\nMEDICATIONS:")
        if entities_dict["MEDICATIONS"]:
            for entity in entities_dict["MEDICATIONS"]:
                code_str = get_code_string(entity)
                print(f"- {entity.get('text', '')}{code_str} [{entity.get('confidence', 0.0):.1f}]")
        else:
            print("- None identified")
        
        # Print conditions/diagnoses if present
        if "CONDITIONS" in entities_dict and entities_dict["CONDITIONS"]:
            print("\nCONDITIONS:")
            for entity in entities_dict["CONDITIONS"]:
                code_str = get_code_string(entity)
                print(f"- {entity.get('text', '')}{code_str} [{entity.get('confidence', 0.0):.1f}]")
        
        # Print procedures if present
        if "PROCEDURES" in entities_dict and entities_dict["PROCEDURES"]:
            print("\nPROCEDURES:")
            for entity in entities_dict["PROCEDURES"]:
                code_str = get_code_string(entity)
                print(f"- {entity.get('text', '')}{code_str} [{entity.get('confidence', 0.0):.1f}]")
        
        # Print lab tests if present
        if "LAB_TESTS" in entities_dict and entities_dict["LAB_TESTS"]:
            print("\nLAB TESTS:")
            for entity in entities_dict["LAB_TESTS"]:
                code_str = get_code_string(entity)
                print(f"- {entity.get('text', '')}{code_str} [{entity.get('confidence', 0.0):.1f}]")
        
        # Print treatment efficacy
        print("\nTREATMENT EFFICACY:")
        if entities_dict["TREATMENT_EFFICACY"]:
            for entity in entities_dict["TREATMENT_EFFICACY"]:
                print(f"- {entity.get('treatment', '')}: {entity.get('efficacy', '')} [{entity.get('confidence', 0.0):.1f}]")
        else:
            print("- None identified")
        
        # Print recommendations
        print("\nRECOMMENDATIONS:")
        if entities_dict["RECOMMENDATIONS"]:
            for entity in entities_dict["RECOMMENDATIONS"]:
                print(f"- {entity.get('text', '')} [{entity.get('confidence', 0.0):.1f}]")
        else:
            print("- None identified")
        
        print("\n======================================\n")


    def process_transcript(self, transcript_data, output_dir=None):
        try:
            start_time = time.time()
        
            # Set default output directory if not provided
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "results")
                os.makedirs(output_dir, exist_ok=True)
                print(f"No output directory provided, using default: {output_dir}")
                



            if isinstance(transcript_data, str):
                # Check if it's a file path or raw text
                if transcript_data.endswith('.json') and os.path.exists(transcript_data):
                    print(f"Processing transcript file: {transcript_data}")
                    # Load from file path
                    with open(transcript_data, 'r') as f:
                        transcript = json.load(f)
                else:
                    # It's raw text, create a simple transcript structure
                    print(f"Processing transcript text: {transcript_data[:50]}...")
                    transcript = {
                        "speaker_turns": [
                            {
                                "speaker": 0,
                                "text": transcript_data
                            }
                        ]
                    }
            elif isinstance(transcript_data, dict):
                # Already loaded data
                print("Processing provided transcript data dictionary")
                transcript = transcript_data
            else:
                raise ValueError("Transcript data must be a file path, raw text, or transcript dictionary")
                    
            # Get full text from transcript
            full_text = ""
            for turn in transcript.get("speaker_turns", []):
                full_text += turn.get("text", "") + " "
                
            print(f"Total transcript length: {len(full_text)} characters")

            # Store original transcript for reference
            transcript = transcript_data

        
            
            # Step 2: Extract medical entities
            print("\nStep 2: Extracting medical entities...")
            entities_time = time.time()
            entities = self.entity_recognizer.extract_entities(full_text)
            print(f"Extracted {len(entities)} entities in {time.time() - entities_time:.2f} seconds")
            
            # Step 2.5: Extract clinical entities
            print("\nStep 2.5: Extracting structured clinical entities...")
            clinical_time = time.time()
            structured_entities = {}
            if hasattr(self, 'clinical_recognizer') and self.clinical_recognizer:
                structured_entities = self.clinical_recognizer.extract_clinical_entities(full_text)
            else:
                print("WARNING: Clinical recognizer not available, using basic entity extraction")
                # Create basic structured entities from regular entities
                structured_entities = self._convert_to_structured_entities(entities)
            
            # Save structured entities
            structured_entities_path = os.path.join(output_dir, "structured_clinical_entities.json")
            with open(structured_entities_path, 'w') as f:
                json.dump(structured_entities, f, indent=2)
            print(f"Saved structured clinical entities to: {structured_entities_path}")
            
            # Step 2.6: Map entities to standard terminology
            print("\nStep 2.6: Mapping entities to terminology...")
            mapping_time = time.time()
            mapped_entities = self.entity_mapper.map_entities(entities)
            print(f"Mapped entities in {time.time() - mapping_time:.2f} seconds")

            # Step 2.7: Resolving medical codes for structured entities...
            print("\nStep 2.7: Resolving medical codes for structured entities...")
            codes_time = time.time()
            resolved_count = 0

            # Process each entity category
            for category, entities_list in structured_entities.items():
                for entity in entities_list:
                    if 'entity_type' in entity and 'text' in entity:
                        entity_type = entity['entity_type']
                        entity_text = entity['text']
                        
                        # Call the terminology resolver
                        resolved = self.terminology_resolver.resolve_entity(entity_text, entity_type)
                        if resolved:
                            # Add individual code fields
                            added_codes = []
                            # Extended list of code types to check for
                            code_types = [
                                'preferred_term', 'rxnorm_code', 'icd10_code', 'snomed_code', 
                                'loinc_code', 'cpt_code', 'hcpcs_code', 'ndc_code'
                            ]
                            
                            for key in code_types:
                                if key in resolved:
                                    entity[key] = resolved[key]
                                    if key.endswith('_code'):  # Only include actual codes
                                        added_codes.append(f"{key}={resolved[key]}")
                            
                            # Transfer any other _code fields not in our standard list
                            for key, value in resolved.items():
                                if key.endswith('_code') and key not in code_types:
                                    entity[key] = value
                                    added_codes.append(f"{key}={value}")
                            
                            # Add a combined "code" field with all codes
                            if added_codes:
                                entity['code'] = ", ".join(added_codes)
                                print(f"  Resolved '{entity_text}' → {entity['code']}")
                                resolved_count += 1
                                
                            # Ensure at least a default code field is present
                            if 'code' not in entity:
                                entity['code'] = ""

            print(f"Resolved medical codes for {resolved_count} entities in {time.time() - codes_time:.2f} seconds")

            # For entries without codes, add defaults based on term with better pattern matching
            print("\nApplying fallback codes for entities without resolved codes...")
            fallback_count = 0
            for category, entities_list in structured_entities.items():
                for entity in entities_list:
                    text = entity.get('text', '').lower()
                    if not entity.get('code') or entity['code'] == "":
                        # Add default codes for known symptoms with partial matching
                        if "headache" in text:
                            entity['icd10_code'] = "R51"
                            entity['snomed_code'] = "25064002"
                            entity['code'] = "icd10_code=R51, snomed_code=25064002"
                            fallback_count += 1
                        elif "migraine" in text:
                            entity['icd10_code'] = "G43.909"
                            entity['snomed_code'] = "37796009"
                            entity['code'] = "icd10_code=G43.909, snomed_code=37796009"
                            fallback_count += 1
                        elif "nausea" in text:
                            entity['icd10_code'] = "R11.0"
                            entity['snomed_code'] = "422587007"
                            entity['code'] = "icd10_code=R11.0, snomed_code=422587007"
                            fallback_count += 1
                        elif "dizzy" in text or "dizziness" in text:
                            entity['icd10_code'] = "R42"
                            entity['snomed_code'] = "404640003"
                            entity['code'] = "icd10_code=R42, snomed_code=404640003"
                            fallback_count += 1
                        elif "vomit" in text:
                            entity['icd10_code'] = "R11.1"
                            entity['snomed_code'] = "422400008"
                            entity['code'] = "icd10_code=R11.1, snomed_code=422400008"
                            fallback_count += 1
                        elif "fever" in text:
                            entity['icd10_code'] = "R50.9"
                            entity['snomed_code'] = "386661006"
                            entity['code'] = "icd10_code=R50.9, snomed_code=386661006"
                            fallback_count += 1
                        elif "fatigue" in text or "tired" in text:
                            entity['icd10_code'] = "R53"
                            entity['snomed_code'] = "84229001"
                            entity['code'] = "icd10_code=R53, snomed_code=84229001"
                            fallback_count += 1
                        elif "pain" in text:
                            entity['icd10_code'] = "R52"
                            entity['snomed_code'] = "22253000"
                            entity['code'] = "icd10_code=R52, snomed_code=22253000"
                            fallback_count += 1
                        elif "shortness of breath" in text or "dyspnea" in text or "breathless" in text:
                            entity['icd10_code'] = "R06.00"
                            entity['snomed_code'] = "267036007"
                            entity['code'] = "icd10_code=R06.00, snomed_code=267036007"
                            fallback_count += 1
                        elif "cough" in text:
                            entity['icd10_code'] = "R05"
                            entity['snomed_code'] = "49727002"
                            entity['code'] = "icd10_code=R05, snomed_code=49727002"
                            fallback_count += 1
                        # Add more medication codes for common treatments
                        elif "sumatriptan" in text:
                            entity['rxnorm_code'] = "10109"
                            entity['code'] = "rxnorm_code=10109"
                            fallback_count += 1
                        elif "ondansetron" in text:
                            entity['rxnorm_code'] = "7804"
                            entity['code'] = "rxnorm_code=7804"
                            fallback_count += 1
                        elif "ibuprofen" in text:
                            entity['rxnorm_code'] = "5640"
                            entity['code'] = "rxnorm_code=5640"
                            fallback_count += 1
                        elif "acetaminophen" in text or "tylenol" in text:
                            entity['rxnorm_code'] = "161"
                            entity['code'] = "rxnorm_code=161"
                            fallback_count += 1
            
            print(f"Applied fallback codes to {fallback_count} entities")

            # Save structured entities with codes to a separate file
            coded_entities_path = os.path.join(output_dir, "structured_clinical_entities_with_codes.json")
            with open(coded_entities_path, 'w') as f:
                json.dump(structured_entities, f, indent=2)
            print(f"Saved structured clinical entities with codes to: {coded_entities_path}")

            # Also save the coded entities to the original path but with a deep copy to ensure no reference issues
            import copy
            with open(structured_entities_path, 'w') as f:
                json.dump(copy.deepcopy(structured_entities), f, indent=2)


            # Step 3: Extract relations
            print("\nStep 3: Extracting relations...")
            relations_time = time.time()
            relations = self.relation_extractor.extract_relations(entities, full_text)
            print(f"Extracted {len(relations)} relations in {time.time() - relations_time:.2f} seconds")
            
            # Step 4: Build knowledge graph
            print("\nStep 4: Building knowledge graph...")
            graph_time = time.time()
            try:
                knowledge_graph = self.graph_builder.build_graph_from_entities_and_relations(mapped_entities, relations)
                print(f"Built knowledge graph in {time.time() - graph_time:.2f} seconds")
            except Exception as e:
                print(f"Error building graph: {e}")
                knowledge_graph = self.graph_builder.build_empty_graph()  # Create an empty graph as fallback
                print("Created an empty fallback graph")
            
            # Step 5: Validate graph
            print("\nStep 5: Validating graph...")
            validation_time = time.time()
            try:
                validation_results = self.graph_validator.validate_graph(self.model_manager, knowledge_graph)
                print(f"Validated graph in {time.time() - validation_time:.2f} seconds")
            except Exception as e:
                print(f"Error validating graph: {e}")
                validation_results = {"valid": False, "errors": [str(e)]}
            
            # Generate output files
            graph_json = self.graph_builder.to_json(knowledge_graph)
            
            # Save results
            graph_path = os.path.join(output_dir, "transcript_graph.json")
            with open(graph_path, 'w') as f:
                json.dump(graph_json, f, indent=2)
                
            # Also save the original entities and relations
            entities_path = os.path.join(output_dir, "entities.json")
            with open(entities_path, 'w') as f:
                json.dump(entities, f, indent=2)
                
            relations_path = os.path.join(output_dir, "relations.json")
            with open(relations_path, 'w') as f:
                json.dump(relations, f, indent=2)
            
            print(f"\nProcessing complete in {time.time() - start_time:.2f} seconds")
            print(f"Results saved to {output_dir}")
            
            # Step 6: Generate clinical notes
            print("\nStep 6: Generating clinical notes...")
            note_time = time.time()

            try:
                # Create note bridge
                note_bridge = NoteBridge()
                
                # Build a result dictionary for the note generator
                note_gen_result = {
                    "transcript": transcript,
                    "structured_entities": structured_entities,
                    "entities": entities,
                    "relations": relations,
                    "graph": graph_json
                }
                
                # Generate the notes
                note_paths = note_bridge.generate_note(note_gen_result, output_dir)
                print(f"Generated clinical notes in {time.time() - note_time:.2f} seconds")
                
                # Then update the return statement to include the note paths
                return {
                    "transcript": transcript,
                    "entities": entities,
                    "structured_entities": structured_entities,
                    "mapped_entities": mapped_entities,
                    "relations": relations,
                    "graph": graph_json,
                    "validation": validation_results,
                    "notes": note_paths  # Add this line to include the note paths
                }
            except Exception as e:
                print(f"Error generating clinical notes: {e}")
                import traceback
                traceback.print_exc()
                
                # If note generation fails, still return the rest of the results
                return {
                    "transcript": transcript,
                    "entities": entities,
                    "structured_entities": structured_entities,
                    "mapped_entities": mapped_entities,
                    "relations": relations,
                    "graph": graph_json,
                    "validation": validation_results
                }

            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in process_transcript: {e}")
            return None
    
    def run(self, transcript_path):
        print(f"Processing file: {transcript_path}")
        try:
            with open(transcript_path, 'r', encoding='utf-8') as file:
                try:
                    # Try to parse as JSON first
                    transcript_data = json.load(file)
                    
                    # Handle structured JSON transcript
                    if isinstance(transcript_data, dict):
                        # Extract text from speaker turns if available
                        transcript_text = ""
                        if "speaker_turns" in transcript_data:
                            for turn in transcript_data["speaker_turns"]:
                                if "text" in turn and turn["text"]:
                                    transcript_text += turn["text"] + " "
                        # If no speaker turns, serialize the whole JSON
                        if not transcript_text:
                            transcript_text = json.dumps(transcript_data)
                    else:
                        transcript_text = str(transcript_data)
                except json.JSONDecodeError:
                    # If not valid JSON, treat as plain text
                    file.seek(0)  # Reset file pointer
                    transcript_text = file.read()
            
            print(f"Processing transcript text: {transcript_text[:100]}...")
            print(f"Total transcript length: {len(transcript_text)} characters")
            
            # Process in smaller chunks if the transcript is very long
            if len(transcript_text) > 5000:
                print("Transcript is long. Processing first 5000 characters for demonstration...")
                transcript_text = transcript_text[:5000]
            
            knowledge_graph = self.process_transcript(transcript_text)
            
            # Save the results
            self.save_results(knowledge_graph, transcript_path)
            
            return knowledge_graph
        except Exception as e:
            print(f"Error processing transcript: {e}")
            traceback.print_exc()
            return None
    
    def save_results(self, result, transcript_path):
        """Save the results to file"""
        try:
            # Create output directory
            output_dir = os.path.join(os.path.dirname(transcript_path), "results")
            os.makedirs(output_dir, exist_ok=True)
            
            # Also save to main results directory
            main_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results")
            os.makedirs(main_output_dir, exist_ok=True)
            
            # Get base filename
            base_name = os.path.splitext(os.path.basename(transcript_path))[0]
            
            # Save entities, relations, and graph
            if "entities" in result:
                with open(os.path.join(output_dir, "entities.json"), 'w') as f:
                    json.dump(result["entities"], f, indent=2)
                with open(os.path.join(main_output_dir, "entities.json"), 'w') as f:
                    json.dump(result["entities"], f, indent=2)
                    
            if "relations" in result:
                with open(os.path.join(output_dir, "relations.json"), 'w') as f:
                    json.dump(result["relations"], f, indent=2)
                with open(os.path.join(main_output_dir, "relations.json"), 'w') as f:
                    json.dump(result["relations"], f, indent=2)
            
            if "graph" in result:
                with open(os.path.join(output_dir, "transcript_graph.json"), 'w') as f:
                    json.dump(result["graph"], f, indent=2)
                with open(os.path.join(main_output_dir, "transcript_graph.json"), 'w') as f:
                    json.dump(result["graph"], f, indent=2)
            
            # Save note paths if available
            if "notes" in result:
                note_paths = result["notes"]
                # Save note paths to a JSON file
                with open(os.path.join(output_dir, "note_paths.json"), 'w') as f:
                    json.dump(note_paths, f, indent=2)
                with open(os.path.join(main_output_dir, "note_paths.json"), 'w') as f:
                    json.dump(note_paths, f, indent=2)
                    
                # Print note generation results
                print("\nGenerated clinical notes:")
                for format_type, path in note_paths.items():
                    print(f"- {format_type.upper()}: {path}")
            
            print(f"Results saved to {output_dir}")
            print(f"Results saved to {main_output_dir}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    # Use command line argument if provided, otherwise use default test file
    transcript_path = sys.argv[1] if len(sys.argv) > 1 else r"data/test_transcripts/transcript.json"
    
    pipeline = MedicalNLPPipeline()
    result = pipeline.run(transcript_path)
    
    if result:
        print("\nPipeline completed successfully.")
        print(f"Generated graph contains {len(result['graph']['nodes'])} nodes and {len(result['graph']['edges'])} edges")
        
        # Display note information if available
        if "notes" in result and result["notes"]:
            print("\nGenerated clinical notes:")
            for format_type, path in result["notes"].items():
                print(f"- {format_type.upper()}: {path}")
    else:
        print("\nPipeline failed to produce results.")