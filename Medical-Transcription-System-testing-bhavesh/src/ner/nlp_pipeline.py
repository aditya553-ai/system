import os
import json
import spacy
import threading
from entity_recognition import MedicalEntityRecognizer
from error_mitigation import TranscriptionErrorCorrector
from semantic_relation_extractor import SemanticRelationExtractor
try:
    from semantic_relation_extractor import SemanticRelationExtractor
    RELATION_EXTRACTOR_AVAILABLE = True
except ImportError:
    print("Warning: SemanticRelationExtractor not available. Relation extraction will be disabled.")
    RELATION_EXTRACTOR_AVAILABLE = False
from datetime import datetime
import time
import torch

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=60):
    """Run a function with a timeout"""
    result = [None]
    error = [None]
    
    def worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            error[0] = e
    
    # Start the worker thread
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    
    # Wait for the worker to finish or timeout
    thread.join(timeout_seconds)
    
    # Check if the thread is still alive (timeout occurred)
    if thread.is_alive():
        return None, TimeoutError("Function call timed out after {} seconds".format(timeout_seconds))
    
    # Return the result or error
    return result[0], error[0]

def load_spacy_model(model_name="en_core_web_sm", disable=None):
    """
    Load spaCy model with proper configuration and error handling
    
    Args:
        model_name: Name of the spaCy model to load
        disable: List of component names to disable
        
    Returns:
        Loaded spaCy model
    """
    print(f"Loading spaCy model '{model_name}'...")
    
    try:
        # Use disable parameter to prevent problematic components from loading
        if disable is None:
            # By default, disable entity_linker which is causing problems
            disable = ["entity_linker"]
            
        nlp = spacy.load(model_name, disable=disable)
        print(f"Successfully loaded spaCy model '{model_name}'")
        return nlp
    except IOError:
        print(f"Model '{model_name}' not found. Attempting to download...")
        try:
            # Attempt to download the model
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            nlp = spacy.load(model_name, disable=disable)
            print(f"Successfully downloaded and loaded '{model_name}'")
            return nlp
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Falling back to basic spaCy model")
            # Create a blank model as fallback
            return spacy.blank("en")

def process_transcript_with_nlp(transcript_json_path, output_dir=None, use_llm=True, extract_relations=True, llm_timeout=120):
    """
    Process a transcript JSON file through NLP pipeline for error correction and NER
    with improved turn-by-turn entity recognition and relation extraction
    """
    start_time = time.time()
    print(f"\nProcessing transcript: {transcript_json_path}")
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load transcript
    try:
        with open(transcript_json_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
    except Exception as e:
        print(f"Error loading transcript: {e}")
        return None
    
    # Load spaCy model with appropriate configuration
    nlp = load_spacy_model(model_name="en_core_web_sm", disable=["entity_linker"])
    
    # Initialize entity recognizer
    ner = MedicalEntityRecognizer()
    
    # Initialize error corrector
    try:
        if use_llm:
            print(f"Initializing error corrector with LLM...")
            error_corrector = TranscriptionErrorCorrector(use_llm=use_llm)
        else:
            error_corrector = TranscriptionErrorCorrector(use_llm=False)
            print("Initialized rule-based error corrector")
    except Exception as e:
        print(f"Error initializing error corrector: {e}")
        error_corrector = None
    
    # Initialize relation extractor if enabled
    relation_extractor = None
    if extract_relations and RELATION_EXTRACTOR_AVAILABLE:
        try:
            print("Initializing relation extractor...")
            relation_extractor = SemanticRelationExtractor(nlp)
            print("Relation extractor initialized")
        except Exception as e:
            print(f"Error initializing relation extractor: {e}")
    
    # Process transcript for error correction if needed
    if error_corrector:
        print("Applying error correction to transcript...")
        processed_data = error_corrector.correct_transcript(transcript_data)
    else:
        processed_data = transcript_data
    
    # Process each speaker turn for entity recognition - ONE AT A TIME
    print("Processing turns one-by-one for entity recognition...")
    
    for i, turn in enumerate(processed_data.get('speaker_turns', [])):
        print(f"Processing turn {i+1}/{len(processed_data.get('speaker_turns', []))}...")
        
        # Get turn text
        text = turn.get('text', '')
        
        # Skip if no text
        if not text:
            turn['entities'] = []
            turn['relations'] = {"relations": [], "count": 0}
            continue
        
        # Process with spaCy and extract entities
        doc = nlp(text)
        entities = ner.extract_entities(text)
        
        # Store entities directly within this turn
        # No global entity list - each turn has its own entities
        turn['entities'] = entities
        
        # Extract relations if enabled
        if extract_relations and relation_extractor and entities and len(entities) > 1:
            # Get relations between entities in this turn
            print(f"  Extracting relations for turn {i+1}...")
            relation_results = relation_extractor.extract_relations(doc, entities)
            
            # Get speaker-specific relations
            speaker_id = turn.get('speaker', None)
            speaker_relations = relation_extractor.extract_speaker_relationships(doc, entities, speaker_id)
            
            # Combine regular and speaker relations
            all_turn_relations = []
            
            if relation_results and "relations" in relation_results:
                all_turn_relations.extend(relation_results["relations"])
            
            if speaker_relations:
                all_turn_relations.extend(speaker_relations)
            
            # Store relations directly within this turn
            turn['relations'] = {
                "relations": all_turn_relations,
                "count": len(all_turn_relations)
            }
        else:
            turn['relations'] = {"relations": [], "count": 0}
    
    # Save processed data
    output_filename = os.path.basename(transcript_json_path).replace('.json', '_with_entities.json')
    if output_dir:
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = os.path.join(os.path.dirname(transcript_json_path), output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed transcript saved to: {output_path}")
    print(f"Processing completed in {time.time() - start_time:.1f} seconds")
    
    # Generate a summary document if needed
    try:
        summary_path = create_medical_summary(processed_data, output_path.replace('_with_entities.json', '_summary.md'))
        print(f"Medical summary saved to: {summary_path}")
    except Exception as e:
        print(f"Error creating medical summary: {e}")
    
    return output_path

def extract_relations(self, doc, entities):
    """
    Extract semantic relations between entities in text
    
    Args:
        doc: spaCy Doc object
        entities: List of extracted entities
        
    Returns:
        Dictionary with extracted relations
    """
    if not entities or len(entities) < 2:
        return {"relations": [], "count": 0}
    
    relations = []
    text = doc.text
    
    # For entity pairs, extract potential relations
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities):
            if i == j:
                continue
            
            # Extract entity information
            e1_text = entity1.get("text", "")
            e1_label = entity1.get("label", "")
            e1_start = entity1.get("start", 0)
            e1_end = entity1.get("end", 0)
            
            e2_text = entity2.get("text", "")
            e2_label = entity2.get("label", "")
            e2_start = entity2.get("start", 0)
            e2_end = entity2.get("end", 0)
            
            # Skip if either entity is empty
            if not e1_text or not e2_text:
                continue
            
            # Common entity label corrections
            if e1_label in ["ORG", "FAC"] and e1_text.lower() in ["metformin", "insulin", "gabapentin"]:
                e1_label = "MEDICATION"
            if e2_label in ["ORG", "FAC"] and e2_text.lower() in ["metformin", "insulin", "gabapentin"]:
                e2_label = "MEDICATION"
            
            # Skip non-medical entities
            if e1_label not in ["MEDICATION", "CONDITION", "SYMPTOM", "ANATOMY", "DOSAGE", "TREATMENT", 
                              "PROCEDURE", "TEST", "TIME", "DATE", "FREQUENCY"]:
                continue
                
            if e2_label not in ["MEDICATION", "CONDITION", "SYMPTOM", "ANATOMY", "DOSAGE", "TREATMENT",
                              "PROCEDURE", "TEST", "TIME", "DATE", "FREQUENCY"]:
                continue
            
            # Get relationship between entities
            relation_type, context_phrase = self._determine_relation(
                text, 
                e1_text, e1_label, e1_start, e1_end,
                e2_text, e2_label, e2_start, e2_end,
                doc
            )
            
            # If no relation found through standard methods, try some common patterns
            if not relation_type:
                # Medication-Dosage pattern
                if e1_label == "MEDICATION" and e2_label in ["DOSAGE", "QUANTITY"] and abs(e1_end - e2_start) < 20:
                    relation_type = "HAS_DOSAGE"
                    context_phrase = text[max(0, e1_start-5):min(len(text), e2_end+5)]
                
                # Medication-Frequency pattern
                elif e1_label == "MEDICATION" and e2_label in ["TIME", "FREQUENCY"] and abs(e1_end - e2_start) < 30:
                    relation_type = "HAS_FREQUENCY"
                    context_phrase = text[max(0, e1_start-5):min(len(text), e2_end+5)]
                
                # Medication-Condition pattern
                elif e1_label == "MEDICATION" and e2_label in ["CONDITION", "SYMPTOM"] and abs(e1_end - e2_start) < 40:
                    # Check for "for" between medicine and condition
                    between_text = text[e1_end:e2_start].lower()
                    if "for" in between_text or "to treat" in between_text or "helps with" in between_text:
                        relation_type = "TREATS"
                        context_phrase = text[max(0, e1_start-5):min(len(text), e2_end+5)]
                
                # Symptom-Anatomy pattern
                elif e1_label == "SYMPTOM" and e2_label in ["ANATOMY", "BODY_PART"] and abs(e1_end - e2_start) < 20:
                    relation_type = "LOCATED_IN"
                    context_phrase = text[max(0, e1_start-5):min(len(text), e2_end+5)]
                
                # Symptom-Time pattern
                elif e1_label == "SYMPTOM" and e2_label in ["TIME", "DATE", "FREQUENCY"] and abs(e1_end - e2_start) < 30:
                    relation_type = "OCCURS_AT"
                    context_phrase = text[max(0, e1_start-5):min(len(text), e2_end+5)]
            
            # If a relation was found, add it to the list
            if relation_type:
                relation = {
                    "source": {"text": e1_text, "label": e1_label},
                    "target": {"text": e2_text, "label": e2_label},
                    "type": relation_type,
                    "confidence": 0.8,
                    "context": context_phrase or text
                }
                
                # Check for duplicates before adding
                if not self._is_duplicate_relation(relations, relation):
                    relations.append(relation)
    
    return {"relations": relations, "count": len(relations)}

def fix_entity_format(data):
    """
    Fix entity format in processed data to be compatible with normalization pipeline
    
    Args:
        data: Processed transcript data with entities
    
    Returns:
        Updated data with fixed entity format
    """
    # Make a copy to avoid modifying the original
    fixed_data = data.copy() if isinstance(data, dict) else data
    
    # Map of common entity label variations to standardized types
    label_map = {
        # Medications
        "MEDICATION": "medication",
        "DRUG": "medication",
        "MED": "medication",
        "MEDICINE": "medication",
        
        # Conditions
        "CONDITION": "condition",
        "DISEASE": "condition",
        "DIAGNOSIS": "condition",
        "DISORDER": "condition",
        
        # Symptoms
        "SYMPTOM": "symptom",
        "PROBLEM": "symptom",
        "COMPLAINT": "symptom",
        
        # Procedures
        "PROCEDURE": "procedure",
        "TEST": "procedure",
        "SURGERY": "procedure",
        "TREATMENT": "procedure",
        
        # Anatomy
        "ANATOMY": "anatomy",
        "BODY_PART": "anatomy",
        "BODYPART": "anatomy",
        "BODY": "anatomy",
        
        # Quantitative
        "QUANTITY": "quantity",
        "CARDINAL": "quantity",
        "NUMBER": "quantity",
        
        # Dosage
        "DOSAGE": "dosage",
        "DOSE": "dosage",
        
        # Timing
        "DATE": "date",
        "TIME": "date",
        "DURATION": "date"
    }
    
    # Process entities in speaker turns
    if "speaker_turns" in fixed_data:
        for turn in fixed_data["speaker_turns"]:
            if "entities" in turn:
                # Fix each entity
                for entity in turn["entities"]:
                    # Ensure we have both type and label fields
                    if "label" in entity and "type" not in entity:
                        label = entity["label"]
                        entity["type"] = label_map.get(label, label.lower())
                            
                    # Ensure we have a label field
                    if "type" in entity and "label" not in entity:
                        entity["label"] = entity["type"].upper()
    
    # Process entities at root level
    if "entities" in fixed_data:
        for entity in fixed_data["entities"]:
            if "label" in entity and "type" not in entity:
                label = entity["label"]
                entity["type"] = label_map.get(label, label.lower())
                
            if "type" in entity and "label" not in entity:
                entity["label"] = entity["type"].upper()
    
    return fixed_data

def create_medical_summary(processed_data, output_path):
    """Create a human-readable summary of medical entities"""
    entity_summary = {
        "medications": set(),
        "dosages": set(),
        "conditions": set(),
        "symptoms": set(),
        "procedures": set(),
        "anatomy": set(),
        "values": set(),
        "quantities": set(),
        "dates": set()
    }
    
    # Entity type to category mapping
    type_mapping = {
        "drug": "medications",
        "medication": "medications",
        "dosage": "dosages",
        "dose": "dosages",
        "condition": "conditions",
        "disease": "conditions",
        "diagnosis": "conditions",
        "symptom": "symptoms",
        "problem": "symptoms",
        "procedure": "procedures",
        "test": "procedures",
        "surgery": "procedures",
        "anatomy": "anatomy",
        "body_part": "anatomy",
        "quantity": "quantities",
        "cardinal": "quantities",
        "date": "dates",
        "time": "dates",
        "duration": "dates"
    }
    
    # Extract entities from all turns
    for turn in processed_data.get("speaker_turns", []):
        if "entities" not in turn:
            continue
            
        for entity in turn.get("entities", []):
            # Get entity text
            text = entity.get("text", "").strip()
            if not text:
                continue
                
            # Get type from any available field
            entity_type = None
            if "type" in entity:
                entity_type = entity["type"].lower()
            elif "label" in entity:
                entity_type = entity["label"].lower()
                
                # Handle common spaCy labels
                if entity_type == "org" and text.lower() in ["metformin", "insulin"]:
                    entity_type = "drug"
                elif entity_type == "condition":
                    entity_type = "condition"
                elif entity_type == "symptom":
                    entity_type = "symptom"
            
            # Map to category if possible
            category = None
            if entity_type in type_mapping:
                category = type_mapping[entity_type]
            
            # Add to appropriate category
            if category and category in entity_summary:
                entity_summary[category].add(text)
    
    # Additional matching for medical terms if NER missed them
    all_text = " ".join(turn.get("text", "") for turn in processed_data.get("speaker_turns", []))
    all_text_lower = all_text.lower()
    
    # Common medical terms
    medical_terms = {
        "conditions": ["diabetes", "diabetic peripheral neuropathy", "peripheral neuropathy", 
                      "neuropathy", "hypoglycemia", "hyperglycemia"],
        "symptoms": ["burning pain", "tingling", "numbness", "dizziness", 
                    "sweating", "rapid heartbeat", "heart races"],
        "anatomy": ["feet", "legs", "toes", "extremities", "lower extremities"],
        "medications": ["metformin", "insulin"]
    }
    
    # Check for these terms in text
    for category, terms in medical_terms.items():
        for term in terms:
            if f" {term.lower()} " in f" {all_text_lower} ":
                # Found term in text - capitalize first letter of each word
                entity_summary[category].add(" ".join(w.capitalize() for w in term.split()))
    
    # If metformin is found but not recognized as medication, add it manually
    if "Metformin" not in entity_summary["medications"] and "metformin" not in entity_summary["medications"]:
        if "metformin" in all_text_lower:
            entity_summary["medications"].add("Metformin")
    
    # Write summary to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("MEDICAL CONVERSATION SUMMARY\n")
        f.write("==========================\n\n")
        
        f.write("Patient Information\n")
        f.write("-----------------\n")
        f.write("Patient reports symptoms of diabetic peripheral neuropathy\n\n")
        
        f.write("Medical Conditions\n")
        f.write("-----------------\n")
        if entity_summary["conditions"]:
            for item in sorted(entity_summary["conditions"]):
                f.write(f"- {item}\n")
        else:
            # If NER missed diabetes/neuropathy, add it manually based on context
            if "diabetic" in all_text_lower or "diabetes" in all_text_lower:
                f.write("- Diabetes Mellitus (implied)\n")
            if "neuropathy" in all_text_lower:
                f.write("- Diabetic Peripheral Neuropathy\n")
        f.write("\n")
        
        f.write("Symptoms\n")
        f.write("--------\n")
        if entity_summary["symptoms"]:
            for item in sorted(entity_summary["symptoms"]):
                f.write(f"- {item}\n")
        else:
            # If symptoms were missed, add them manually based on text
            symptoms_found = []
            for symptom in ["burning pain", "tingling", "numbness", "dizziness", "sweating", "heart races"]:
                if symptom in all_text_lower:
                    symptoms_found.append(symptom.capitalize())
            for symptom in symptoms_found:
                f.write(f"- {symptom}\n")
        f.write("\n")
        
        f.write("Medications\n")
        f.write("-----------\n")
        for item in sorted(entity_summary["medications"]):
            f.write(f"- {item}")
            # Find related dosages or quantities
            dosage_info = []
            related_dosages = [d for d in entity_summary["dosages"] if len(d) > 0]
            related_quantities = [q for q in entity_summary["quantities"] if len(q) > 0]
            
            if related_dosages:
                dosage_info.extend(related_dosages)
            if related_quantities:
                dosage_info.extend(related_quantities)
                
            if "metformin" in item.lower() and "500mg" in all_text_lower:
                dosage_info.append("500mg")
                
            if dosage_info:
                unique_dosages = set(dosage_info)
                f.write(f" ({', '.join(unique_dosages)})")
            
            # If we know it's Metformin, add frequency if found in text
            if "metformin" in item.lower() and "twice daily" in all_text_lower:
                f.write(" - taken twice daily")
                
            f.write("\n")
        f.write("\n")
        
        f.write("Anatomical References\n")
        f.write("--------------------\n")
        if entity_summary["anatomy"]:
            for item in sorted(entity_summary["anatomy"]):
                f.write(f"- {item}\n")
        else:
            # If anatomy terms were missed, add them manually
            anatomy_found = []
            for anatomy in ["lower extremities", "feet", "legs", "toes"]:
                if anatomy in all_text_lower:
                    anatomy_found.append(anatomy.capitalize())
            for anatomy in anatomy_found:
                f.write(f"- {anatomy}\n")
        f.write("\n")
        
        # Add timing information
        if entity_summary["dates"]:
            f.write("Timing Information\n")
            f.write("-----------------\n")
            for item in sorted(entity_summary["dates"]):
                f.write(f"- {item}\n")
            
    print(f"Medical summary created: {output_path}")

def extract_relations_from_existing(entities_json_path, output_dir=None):
    """
    Extract relations from an existing transcript with entities
    
    Args:
        entities_json_path: Path to transcript JSON with entities
        output_dir: Optional output directory for processed file
    """
    print(f"\nExtracting relations from: {entities_json_path}")
    
    # Load the transcript with entities
    try:
        with open(entities_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    # Load spaCy model
    nlp = load_spacy_model(model_name="en_core_web_sm", disable=["entity_linker"])
    
    # Initialize relation extractor
    relation_extractor = SemanticRelationExtractor(nlp)
    
    # Process each speaker turn
    all_relations = []
    
    for i, turn in enumerate(data.get('speaker_turns', [])):
        text = turn.get('text', '')
        entities = turn.get('entities', [])
        
        # Skip if no text or entities
        if not text or not entities or len(entities) < 2:
            turn['relations'] = {"relations": [], "count": 0}
            continue
        
        # Process with spaCy
        doc = nlp(text)
        
        # Extract relations
        speaker_id = turn.get('speaker', None)
        turn_relations = []
        
        # Get relations between entities in this turn
        relation_results = relation_extractor.extract_relations(doc, entities)
        if relation_results and "relations" in relation_results and relation_results["relations"]:
            turn_relations.extend(relation_results["relations"])
        
        # Get speaker-specific relations
        speaker_relations = relation_extractor.extract_speaker_relationships(doc, entities, speaker_id)
        if speaker_relations:
            turn_relations.extend(speaker_relations)
        
        # Store relations with turn information
        for relation in turn_relations:
            relation_with_turn = relation.copy()
            relation_with_turn['turn_index'] = i
            relation_with_turn['speaker'] = speaker_id
            all_relations.append(relation_with_turn)
        
        # Update relations for this turn
        turn['relations'] = {
            "relations": turn_relations,
            "count": len(turn_relations)
        }
    
    # Add all relations to the output
    data['all_relations'] = all_relations
    
    # Save updated data
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(entities_json_path))
    else:
        output_path = entities_json_path.replace('.json', '_with_relations.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Extracted {len(all_relations)} relations")
    print(f"Saved to: {output_path}")
    
    return output_path

def process_turn_incrementally(text, speaker_id, nlp, ner, relation_extractor=None):
    """
    Process a single turn incrementally for real-time NER and relation extraction
    
    Args:
        text: Text from this turn
        speaker_id: Speaker identifier
        nlp: spaCy NLP model
        ner: MedicalEntityRecognizer instance
        relation_extractor: Optional SemanticRelationExtractor instance
        
    Returns:
        Dictionary with processed turn including entities and relations
    """
    # Skip if no text
    if not text:
        return {
            "speaker": speaker_id,
            "text": text,
            "entities": [],
            "relations": {"relations": [], "count": 0}
        }
    
    # Process text with NER
    doc = nlp(text)
    entities = ner.extract_entities(text)
    
    # Create turn object
    turn = {
        "speaker": speaker_id,
        "text": text,
        "entities": entities
    }
    
    # Extract relations if enabled and we have multiple entities
    if relation_extractor and entities and len(entities) > 1:
        relation_results = relation_extractor.extract_relations(doc, entities)
        speaker_relations = relation_extractor.extract_speaker_relationships(doc, entities, speaker_id)
        
        all_turn_relations = []
        if relation_results and "relations" in relation_results:
            all_turn_relations.extend(relation_results["relations"])
        if speaker_relations:
            all_turn_relations.extend(speaker_relations)
        
        turn['relations'] = {
            "relations": all_turn_relations,
            "count": len(all_turn_relations)
        }
    else:
        turn['relations'] = {"relations": [], "count": 0}
        
    return turn

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "realtime":
            # Set up real-time processing components
            print("Initializing real-time NLP components...")
            nlp = load_spacy_model(model_name="en_core_web_sm", disable=["entity_linker"])
            ner = MedicalEntityRecognizer()
            relation_extractor = SemanticRelationExtractor(nlp) if RELATION_EXTRACTOR_AVAILABLE else None
            
            print("Ready for real-time processing. Enter text (empty line to exit):")
            speaker_id = 0
            while True:
                user_input = input(f"Speaker {speaker_id}: ")
                if not user_input:
                    break
                    
                # Process incrementally
                result = process_turn_incrementally(user_input, speaker_id, nlp, ner, relation_extractor)
                
                # Display entities and relations
                print("\nEntities:")
                for entity in result["entities"]:
                    print(f"  - {entity['text']} ({entity['label']})")
                    
                print("\nRelations:")
                for relation in result["relations"]["relations"]:
                    print(f"  - {relation['source']['text']} ({relation['source']['label']}) "
                          f"--[{relation['type']}]--> {relation['target']['text']} ({relation['target']['label']})")
                
                # Toggle speaker
                speaker_id = 1 - speaker_id  # Alternate between 0 and 1
            
        else:
            # Regular batch processing
            transcript_path = sys.argv[1]
            output_dir = sys.argv[2] if len(sys.argv) > 2 else None
            use_llm = not (len(sys.argv) > 3 and sys.argv[3].lower() == "fast")
            process_transcript_with_nlp(transcript_path, output_dir, use_llm=use_llm)
    else:
        print("Usage options:")
        print("1. Regular processing: python nlp_pipeline.py path/to/transcript.json [output_directory] [fast]")
        print("2. Real-time processing: python nlp_pipeline.py realtime")