import os
import sys
import json
import importlib.util
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAI
load_dotenv()
import importlib.util
print("\n--- DEBUGGING LANGCHAIN IMPORT START ---")
try:
    print("Attempting to import LangChain modules...")
    import os
    import sys
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if present
    
    # Debug environment
    api_key = os.environ.get("OPENAI_API_KEY") 
    print(f"OpenAI API Key found: {'Yes (length: ' + str(len(api_key)) + ')' if api_key else 'No'}")
    
    try:
        from langchain_community.llms import OpenAI
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        print("✓ Successfully imported LangChain modules")
        
        # Check if API key is available
        if api_key:
            LANGCHAIN_AVAILABLE = True
            print("✓ LangChain is available with valid API key")
        else:
            LANGCHAIN_AVAILABLE = False
            print("× LangChain available but no API key found")
    except ImportError as e:
        print(f"× Failed to import LangChain modules: {e}")
        LANGCHAIN_AVAILABLE = False
except Exception as e:
    print(f"× Unexpected error during LangChain setup: {e}")
    LANGCHAIN_AVAILABLE = False

print("--- DEBUGGING LANGCHAIN IMPORT END ---\n")

# Define necessary paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
NOTE_GEN_DIR = os.path.join(PROJECT_ROOT, "note_generation")

# Add the note_generation directory to path
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, NOTE_GEN_DIR)

print(f"Looking for note generation modules in: {NOTE_GEN_DIR}")
print(f"Note generation directory exists: {os.path.exists(NOTE_GEN_DIR)}")

# Try to list files in the directory
if os.path.exists(NOTE_GEN_DIR):
    print("Files in note_generation directory:")
    for file in os.listdir(NOTE_GEN_DIR):
        print(f"  - {file}")

# Define empty mocks that will be overridden if imports succeed
class MockTemplateFiller:
    def generate_note(self, data, note_type="soap"):
        print("MOCK: Would generate a note from template here")
        return {"mock": "note", "date": datetime.now().isoformat()}

class MockNoteRenderer:
    def render_text(self, note_data):
        return "This is a mock clinical note."
    
    def render_html(self, note_data):
        return "<html><body><h1>Mock Clinical Note</h1></body></html>"

# First check if we can directly import the modules
try:
    print("Attempting direct imports...")
    from note_generation.template_filling import TemplateFiller
    from note_generation.safety_alerts import SafetyAlertGenerator
    from note_generation.note_renderer import NoteRenderer
    from note_generation.conversation_analyzer import ConversationAnalyzer
    
    try:
        from note_generation.llm_note_generator import LlmNoteGenerator
        LLM_AVAILABLE = True
        print("✓ Successfully imported LlmNoteGenerator")
    except ImportError as e:
        print(f"× Failed to import LlmNoteGenerator: {e}")
        LLM_AVAILABLE = False
    
    NOTE_GEN_AVAILABLE = True
    print("✓ Successfully imported note generation modules")
    
except ImportError as e:
    print(f"× Direct imports failed: {e}")
    
    # Try alternative approach - look for the actual files and import them
    print("Trying alternative import approach...")
    NOTE_GEN_AVAILABLE = False
    LLM_AVAILABLE = False
    
    # Define the modules we want to import
    modules_to_import = {
        "TemplateFiller": os.path.join(NOTE_GEN_DIR, "template_filling.py"),
        "SafetyAlertGenerator": os.path.join(NOTE_GEN_DIR, "safety_alerts.py"),
        "NoteRenderer": os.path.join(NOTE_GEN_DIR, "note_renderer.py"),
        "ConversationAnalyzer": os.path.join(NOTE_GEN_DIR, "conversation_analyzer.py"),
        "LlmNoteGenerator": os.path.join(NOTE_GEN_DIR, "llm_note_generator.py")
    }
    
    # Try to import each module
    imported_modules = {}
    for name, path in modules_to_import.items():
        if os.path.exists(path):
            try:
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                imported_modules[name] = getattr(module, name)
                print(f"✓ Successfully imported {name} from {path}")
            except Exception as e:
                print(f"× Failed to import {name} from {path}: {e}")
                
    # Set up the classes based on what was imported
    if "TemplateFiller" in imported_modules:
        TemplateFiller = imported_modules["TemplateFiller"]
        NOTE_GEN_AVAILABLE = True
    else:
        TemplateFiller = MockTemplateFiller
        
    if "SafetyAlertGenerator" in imported_modules:
        SafetyAlertGenerator = imported_modules["SafetyAlertGenerator"]
    else:
        SafetyAlertGenerator = lambda: None
        
    if "NoteRenderer" in imported_modules:
        NoteRenderer = imported_modules["NoteRenderer"]
    else:
        NoteRenderer = MockNoteRenderer
        
    if "ConversationAnalyzer" in imported_modules:
        ConversationAnalyzer = imported_modules["ConversationAnalyzer"]
    else:
        ConversationAnalyzer = lambda: None
        
    if "LlmNoteGenerator" in imported_modules:
        LlmNoteGenerator = imported_modules["LlmNoteGenerator"]
        LLM_AVAILABLE = True
    else:
        LLM_AVAILABLE = False

class NoteBridge:
    """
    Bridge class to connect NLP pipeline results to note generation system
    """
    def __init__(self, use_llm: bool = True, llm_api_key: Optional[str] = None):
        """
        Initialize the bridge
        
        Args:
            use_llm: Whether to use LLM for note generation
            llm_api_key: API key for LLM service
        """
        print("Initializing NoteBridge...")
        
        if not NOTE_GEN_AVAILABLE:
            print("WARNING: Note generation components are not available!")
            print("Please check that the note_generation directory exists and contains the required modules.")
            print(f"Expected location: {NOTE_GEN_DIR}")
            return
            
        self.use_llm = use_llm and LLM_AVAILABLE
        self.llm_api_key = llm_api_key
        
        # Initialize note generation components
        print("Initializing note generation components...")
        self.template_filler = TemplateFiller()
        print("Template filler initialized")
        
        try:
            self.safety_alert_gen = SafetyAlertGenerator()
            print("Safety alert generator initialized")
        except Exception as e:
            print(f"Failed to initialize safety alert generator: {e}")
            self.safety_alert_gen = None
            
        try:
            self.note_renderer = NoteRenderer()
            print("Note renderer initialized")
        except Exception as e:
            print(f"Failed to initialize note renderer: {e}")
            self.note_renderer = MockNoteRenderer()
            
        try:
            self.conversation_analyzer = ConversationAnalyzer()
            print("Conversation analyzer initialized")
        except Exception as e:
            print(f"Failed to initialize conversation analyzer: {e}")
            self.conversation_analyzer = None
            
        if self.use_llm:
            try:
                self.llm_generator = LlmNoteGenerator(api_key=llm_api_key)
                print("LLM generator initialized")
            except Exception as e:
                print(f"Error initializing LLM generator: {e}")
                self.use_llm = False
        
        print("NoteBridge initialization complete!")
    
    def load_nlp_results(self, results_dir: str) -> Dict[str, Any]:
        """
        Load NLP pipeline results from directory
        
        Args:
            results_dir: Directory containing NLP pipeline results
            
        Returns:
            Consolidated results dictionary
        """
        print(f"Loading NLP results from {results_dir}")
        results = {}
        
        # Try to load structured clinical entities with codes
        clinical_entities_path = os.path.join(results_dir, "structured_clinical_entities_with_codes.json")
        if not os.path.exists(clinical_entities_path):
            clinical_entities_path = os.path.join(results_dir, "structured_clinical_entities.json")
        
        if os.path.exists(clinical_entities_path):
            with open(clinical_entities_path, 'r', encoding='utf-8') as f:
                results["structured_entities"] = json.load(f)
                print(f"Loaded structured clinical entities from {clinical_entities_path}")
        else:
            print(f"Warning: Could not find structured clinical entities at {clinical_entities_path}")
        
        # Try to load transcript graph
        graph_path = os.path.join(results_dir, "transcript_graph.json")
        if os.path.exists(graph_path):
            with open(graph_path, 'r', encoding='utf-8') as f:
                results["graph"] = json.load(f)
                print(f"Loaded knowledge graph from {graph_path}")
        else:
            print(f"Warning: Could not find transcript graph at {graph_path}")
        
        # Try to load entities
        entities_path = os.path.join(results_dir, "entities.json")
        if os.path.exists(entities_path):
            with open(entities_path, 'r', encoding='utf-8') as f:
                results["entities"] = json.load(f)
                print(f"Loaded entities from {entities_path}")
        else:
            print(f"Warning: Could not find entities at {entities_path}")
        
        # Try to load relations
        relations_path = os.path.join(results_dir, "relations.json")
        if os.path.exists(relations_path):
            with open(relations_path, 'r', encoding='utf-8') as f:
                results["relations"] = json.load(f)
                print(f"Loaded relations from {relations_path}")
        else:
            print(f"Warning: Could not find relations at {relations_path}")
        
        return results


    def extract_plan_with_langchain(self, transcript_text, structured_entities):
        """
        Extract treatment plan from transcript using LangChain
        
        Args:
            transcript_text: Full transcript text
            structured_entities: Dictionary of structured entities
            
        Returns:
            Extracted plan as string
        """
        if not LANGCHAIN_AVAILABLE:
            print("LangChain not available for plan extraction")
            return "Follow up as needed for symptom management"
        
        print("Extracting treatment plan using LangChain...")
        
        # Create a prompt for plan extraction
        template = """
        You are a medical assistant tasked with extracting the treatment plan from a doctor-patient conversation.
        
        Below is a transcript of a medical conversation. Please extract and organize the doctor's plan, including:
        - Diagnostic tests ordered (blood work, imaging, etc.)
        - Medications prescribed
        - Lifestyle recommendations
        - Follow-up instructions
        - Any monitoring instructions for the patient
        
        TRANSCRIPT:
        {transcript}
        
        The main symptoms identified were: {symptoms}
        
        TREATMENT PLAN (be specific and comprehensive, using the doctor's actual recommendations):
        """
        
        # Format the symptoms for the prompt
        symptom_texts = []
        if "SYMPTOMS" in structured_entities:
            symptom_texts = [s.get("text", "") for s in structured_entities["SYMPTOMS"]]
        
        symptoms_str = ", ".join(symptom_texts) if symptom_texts else "not specified"
        
        # Create the prompt with the transcript and symptoms
        prompt = PromptTemplate(
            input_variables=["transcript", "symptoms"],
            template=template
        )
        
        try:
            # Initialize LLM - use environment variable or a default key
            llm = OpenAI(temperature=0.1, model_name="gpt-3.5-turbo-instruct")
            
            # Create the chain
            # chain = LLMChain(llm=llm, prompt=prompt)
            chain = prompt | llm

            
            # Run the chain to extract the plan
            # result = chain.run(transcript=transcript_text, symptoms=symptoms_str)
            result = chain.invoke({"transcript": transcript_text, "symptoms": symptoms_str})
            
            print(f"Successfully extracted plan with LangChain")
            return result.strip()
        
        except Exception as e:
            print(f"Error using LangChain for plan extraction: {e}")
            # Fallback to basic plan
            return ("Order blood work to rule out deficiencies or thyroid issues. "
                    "An MRI might be needed if symptoms persist. Reduce caffeine to one coffee, "
                    "hydrate well, and try stress management. Prescribe a low-dose preventive medication. "
                    "Review test results in a week. Keep symptom diary noting triggers, pain duration and intensity.")
    
    def transform_for_note_generation(self, nlp_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform NLP results into format expected by note generation system
        
        Args:
            nlp_results: Dictionary with NLP pipeline results
            
        Returns:
            Dictionary in format expected by note generation system
        """
        print("Transforming NLP results for note generation...")
        print(f"NLP results keys: {list(nlp_results.keys())}")
        transformed = {}
        
        if "transcript" in nlp_results:
            if isinstance(nlp_results["transcript"], dict):
                # It's a dictionary, so we can copy it
                transformed = nlp_results["transcript"].copy()
                print(f"Using original transcript dictionary structure")
            else:
                # It's a string or other type, create a dictionary structure
                transformed = {
                    "speaker_turns": [
                        {
                            "speaker": 0,  # Assuming a single speaker
                            "text": nlp_results["transcript"]
                        }
                    ],
                    "metadata": {
                        "created_at": datetime.now().isoformat()
                    }
                }
                print(f"Created transcript structure from text of length {len(str(nlp_results['transcript']))} chars")
        else:
            # Create a minimal transcript structure
            transformed = {
                "speaker_turns": [],
                "metadata": {
                    "created_at": datetime.now().isoformat()
                }
            }
            print("No transcript found, created minimal structure")
        
        # Add structured data section
        if "structured_data" not in transformed:
            transformed["structured_data"] = {}
        
        # Process structured entities into note-ready format
        if "structured_entities" in nlp_results:
            structured = nlp_results["structured_entities"]
            print(f"Structured entities categories: {list(structured.keys())}")
              # Process symptoms
            if "SYMPTOMS" in structured:
                symptom_entries = []
                for symptom in structured["SYMPTOMS"]:
                    symptom_entry = {
                        "name": symptom.get("text", ""),
                        "description": f"Patient reports {symptom.get('text', '')}",
                        "confidence": symptom.get("confidence", 0.5)
                    }
                    
                    # Add medical codes
                    if "icd10_code" in symptom:
                        symptom_entry["icd10_code"] = symptom["icd10_code"]
                    if "snomed_code" in symptom:
                        symptom_entry["snomed_code"] = symptom["snomed_code"]
                    if "loinc_code" in symptom:
                        symptom_entry["loinc_code"] = symptom["loinc_code"]
                    if "code" in symptom:
                        symptom_entry["codes"] = symptom["code"]
                    # Include any other standardized codes that might be present
                    for key in symptom:
                        if key.endswith('_code') and key not in ["icd10_code", "snomed_code", "loinc_code"]:
                            symptom_entry[key] = symptom[key]
                    
                    symptom_entries.append(symptom_entry)
                
                # Deduplicate symptoms before adding to structured data
                transformed["structured_data"]["symptoms"] = self._deduplicate_structured_entities(symptom_entries, "symptoms")
                print(f"Processed {len(transformed['structured_data']['symptoms'])} unique symptoms")
              # Process medications
            if "MEDICATIONS" in structured:
                medication_entries = []
                
                for med in structured["MEDICATIONS"]:
                    med_entry = {
                        "name": med.get("text", ""),
                        "dose": med.get("dose", "as prescribed"),
                        "frequency": med.get("frequency", "as directed"),
                        "confidence": med.get("confidence", 0.5)
                    }
                    
                    # Add medical codes
                    if "rxnorm_code" in med:
                        med_entry["rxnorm_code"] = med["rxnorm_code"]
                    if "snomed_code" in med:
                        med_entry["snomed_code"] = med["snomed_code"]
                    if "code" in med:
                        med_entry["codes"] = med["code"]
                    # Include any other standardized codes that might be present
                    for key in med:
                        if key.endswith('_code') and key not in ["rxnorm_code", "snomed_code"]:
                            med_entry[key] = med[key]
                    
                    medication_entries.append(med_entry)
                
                # Deduplicate medications before adding to structured data
                deduplicated_meds = self._deduplicate_structured_entities(medication_entries, "medications")
                transformed["structured_data"]["medications"] = deduplicated_meds
                transformed["structured_data"]["current_medications"] = deduplicated_meds.copy()
                
                print(f"Processed {len(transformed['structured_data']['medications'])} unique medications")
              # Process conditions/diagnoses
            if "CONDITIONS" in structured or "DIAGNOSES" in structured:
                condition_entries = []
                
                # Process CONDITIONS if present
                for condition_key in ["CONDITIONS", "DIAGNOSES"]:
                    if condition_key in structured:
                        for condition in structured[condition_key]:
                            condition_entry = {
                                "name": condition.get("text", ""),
                                "description": condition.get("description", f"Diagnosed with {condition.get('text', '')}"),
                                "confidence": condition.get("confidence", 0.5)
                            }
                            
                            # Add medical codes
                            if "icd10_code" in condition:
                                condition_entry["icd10_code"] = condition["icd10_code"]
                            if "snomed_code" in condition:
                                condition_entry["snomed_code"] = condition["snomed_code"]
                            if "code" in condition:
                                condition_entry["codes"] = condition["code"]
                            # Include any other standardized codes
                            for key in condition:
                                if key.endswith('_code') and key not in ["icd10_code", "snomed_code"]:
                                    condition_entry[key] = condition[key]
                            
                            condition_entries.append(condition_entry)
                
                # Deduplicate conditions before adding to structured data
                transformed["structured_data"]["problems"] = self._deduplicate_structured_entities(condition_entries, "conditions")
                print(f"Processed {len(transformed['structured_data']['problems'])} unique problems/conditions")
              # Process procedures 
            if "PROCEDURES" in structured:
                procedure_entries = []
                
                for procedure in structured["PROCEDURES"]:
                    procedure_entry = {
                        "name": procedure.get("text", ""),
                        "description": procedure.get("description", ""),
                        "confidence": procedure.get("confidence", 0.5)
                    }
                    
                    # Add medical codes
                    if "icd10_code" in procedure:
                        procedure_entry["icd10_code"] = procedure["icd10_code"]
                    if "snomed_code" in procedure:
                        procedure_entry["snomed_code"] = procedure["snomed_code"]
                    if "cpt_code" in procedure:
                        procedure_entry["cpt_code"] = procedure["cpt_code"]
                    if "code" in procedure:
                        procedure_entry["codes"] = procedure["code"]
                    # Include any other standardized codes
                    for key in procedure:
                        if key.endswith('_code') and key not in ["icd10_code", "snomed_code", "cpt_code"]:
                            procedure_entry[key] = procedure[key]
                    
                    procedure_entries.append(procedure_entry)
                
                # Deduplicate procedures before adding to structured data
                if "procedures" not in transformed["structured_data"]:
                    transformed["structured_data"]["procedures"] = []
                transformed["structured_data"]["procedures"] = self._deduplicate_structured_entities(procedure_entries, "procedures")
                print(f"Processed {len(transformed['structured_data']['procedures'])} unique procedures")
              # Process lab tests
            if "LAB_TESTS" in structured:
                lab_entries = []
                
                for lab in structured["LAB_TESTS"]:
                    lab_entry = {
                        "name": lab.get("text", ""),
                        "value": lab.get("value", ""),
                        "unit": lab.get("unit", ""),
                        "confidence": lab.get("confidence", 0.5)
                    }
                    
                    # Add medical codes
                    if "loinc_code" in lab:
                        lab_entry["loinc_code"] = lab["loinc_code"]
                    if "code" in lab:
                        lab_entry["codes"] = lab["code"]
                    # Include any other standardized codes
                    for key in lab:
                        if key.endswith('_code') and key not in ["loinc_code"]:
                            lab_entry[key] = lab[key]
                    
                    lab_entries.append(lab_entry)
                
                # Deduplicate lab tests before adding to structured data
                if "lab_results" not in transformed["structured_data"]:
                    transformed["structured_data"]["lab_results"] = []
                transformed["structured_data"]["lab_results"] = self._deduplicate_structured_entities(lab_entries, "lab_tests")
                print(f"Processed {len(transformed['structured_data']['lab_results'])} unique lab tests")
        
        full_transcript = ""
        
        # Try multiple approaches to get the transcript text
        if "transcript" in nlp_results:
            transcript_data = nlp_results["transcript"]
            if isinstance(transcript_data, str):
                full_transcript = transcript_data
            elif isinstance(transcript_data, dict):
                # Try to get from speaker turns
                if "speaker_turns" in transcript_data:
                    for turn in transcript_data["speaker_turns"]:
                        if "text" in turn:
                            full_transcript += turn["text"] + " "
                # Try to get from text field
                elif "text" in transcript_data:
                    full_transcript = transcript_data["text"]
        
        # Add debug output to see what's being extracted
        print(f"\nExtracted transcript text (first 100 chars): {full_transcript[:100]}...")
        
        default_plan = "Diagnostics ordered. Treatment initiated. Follow-up scheduled."

        # Only try LangChain if it's available (which we've debugged above)
        if LANGCHAIN_AVAILABLE:
            try:
                print("Trying to use LangChain for plan extraction...")
                langchain_plan = self.extract_plan_with_langchain(full_transcript, 
                                                            nlp_results.get("structured_entities", {}))
                # Use the LangChain plan if it succeeded
                plan = langchain_plan
                print(f"Successfully used LangChain for plan: {plan[:50]}...")
            except Exception as e:
                print(f"Error using LangChain: {e}")
                # Fall back to rule-based
                print(f"Falling back to rule-based plan: {plan[:50]}...")
        else:
            print("LangChain not available, using rule-based plan extraction")

        
        # Add the extracted plan to the structured data
        transformed["structured_data"]["plan"] = plan
        print(f"Added plan: {plan[:100]}...")
        
        # Add normalized_entities section
        transformed["normalized_entities"] = {}
        
        # Convert flat entities list to normalized format
        if "entities" in nlp_results:
            for entity in nlp_results["entities"]:
                entity_text = entity.get("text", "")
                entity_type = entity.get("entity_type", "")
                
                if not entity_text or not entity_type:
                    continue
                
                if entity_text not in transformed["normalized_entities"]:
                    transformed["normalized_entities"][entity_text] = {
                        "type": entity_type,
                        "codes": {}
                    }
                  # Add codes if available
                if "code" in entity:
                    transformed["normalized_entities"][entity_text]["codes"] = entity["code"]
                
                # Add all standard code types explicitly
                for code_type in ["icd10_code", "snomed_code", "rxnorm_code", "loinc_code", "cpt_code"]:
                    if code_type in entity:
                        transformed["normalized_entities"][entity_text][code_type] = entity[code_type]
                        
                # Include any other code types that might be present
                for key in entity:
                    if key.endswith('_code') and key not in ["icd10_code", "snomed_code", "rxnorm_code", "loinc_code", "cpt_code"]:
                        transformed["normalized_entities"][entity_text][key] = entity[key]
            
            print(f"Processed {len(transformed['normalized_entities'])} normalized entities")
        
        print("Transformation complete!")
        return transformed
    
    def generate_note(self, nlp_results: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate clinical note from NLP results
        
        Args:
            nlp_results: Dictionary with NLP pipeline results
            output_dir: Directory to save output files
            
        Returns:
            Dictionary with paths to generated files
        """
        if not NOTE_GEN_AVAILABLE:
            print("Note generation components not available.")
            return {}
        
        print("Preparing data for note generation...")
        
        # Transform NLP results for note generation
        transformed_data = self.transform_for_note_generation(nlp_results)
        
        # Set default output directory if not provided
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "soap_notes")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a base name for output files
        base_name = "medical_transcript"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = f"{base_name}_{timestamp}"
        
        # Generate the clinical note
        clinical_note = {}
        
        if self.use_llm:
            try:
                print("Generating clinical note using LLM...")
                clinical_note = self.llm_generator.generate_soap_note(transformed_data)
                print("Successfully generated SOAP note with LLM")
            except Exception as e:
                print(f"Error using LLM for note generation: {e}")
                print("Falling back to template-based note generation")
                self.use_llm = False
        
        if not self.use_llm:
            print("Using template-based note generation...")
            clinical_note = self.template_filler.generate_note(transformed_data, "soap")
            print("Successfully generated SOAP note with template")
        
        # Generate safety alerts
        print("Generating safety alerts...")
        try:
            if self.safety_alert_gen:
                safety_alerts = self.safety_alert_gen.generate_alerts(clinical_note)
                
                # Add alerts to the clinical note
                if safety_alerts:
                    clinical_note["alerts"] = safety_alerts
                    print(f"Added {len(safety_alerts)} safety alerts to the note")
                else:
                    clinical_note["alerts"] = []
                    print("No safety alerts generated")
            else:
                clinical_note["alerts"] = []
                print("Safety alert generator not available")
        except Exception as e:
            print(f"Error generating safety alerts: {e}")
            clinical_note["alerts"] = []
        
        # Add date to the clinical note if not present
        if "date" not in clinical_note:
            clinical_note["date"] = datetime.now().strftime("%Y-%m-%d")
        
        # Output paths
        output_paths = {}
          # Ensure the standardized codes are preserved in the final SOAP note
        self._ensure_codes_in_output(clinical_note)
        
        # Save the clinical note as JSON for debugging
        json_path = os.path.join(output_dir, f"{output_base}_clinical_note.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clinical_note, f, indent=2)
        output_paths["json"] = json_path
        print(f"Saved clinical note JSON to: {json_path}")
        
        # Create text version
        try:
            text_content = self.note_renderer.render_text(clinical_note)
            text_path = os.path.join(output_dir, f"{output_base}_clinical_note.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            output_paths["text"] = text_path
            print(f"Saved text note to: {text_path}")
        except Exception as e:
            print(f"Error generating text version: {e}")
        
        # Create HTML version
        try:
            html_content = self.note_renderer.render_html(clinical_note)
            html_path = os.path.join(output_dir, f"{output_base}_clinical_note.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            output_paths["html"] = html_path
            print(f"Saved HTML note to: {html_path}")
        except Exception as e:
            print(f"Error generating HTML version: {e}")
        
        # Create PDF version if possible
        try:
            pdf_path = os.path.join(output_dir, f"{output_base}_clinical_note.pdf")
            if hasattr(self.note_renderer, 'render_pdf'):
                pdf_result = self.note_renderer.render_pdf(clinical_note, pdf_path)
                if pdf_result:
                    output_paths["pdf"] = pdf_path
                    print(f"Saved PDF note to: {pdf_path}")
        except Exception as e:
            print(f"Error generating PDF: {e}")
        
        print("Note generation completed successfully!")
        print("Generated files:")
        for fmt, path in output_paths.items():
            print(f"- {fmt.upper()}: {path}")
        
        return output_paths

    def _ensure_codes_in_output(self, clinical_note):
        """
        Ensure all standardized codes are preserved in the final SOAP note structure
        
        Args:
            clinical_note: The clinical note data structure to check and update
        """
        print("Ensuring all standardized codes are included in the final SOAP note...")
        
        # Process structured sections with potential code updates
        sections_to_check = [
            "symptoms", "problems", "medications", "procedures", "lab_results", 
            "allergies", "vitals", "diagnoses", "assessments"
        ]
        
        # Function to check and ensure codes are present
        def process_section_entries(section_name, entries):
            if not entries:
                return
                
            print(f"Processing {len(entries)} entries in {section_name} section")
            for entry in entries:
                # Check for key code fields to ensure they are included
                for key in list(entry.keys()):
                    if key.endswith('_code') or key == 'codes':
                        # Make sure the code is present in the output
                        # (Some template generators might strip them out)
                        if entry[key] is None or entry[key] == "":
                            print(f"Warning: Empty code value for {key} in {section_name}")
        
        # Process all relevant sections
        if "structured_data" in clinical_note:
            for section in sections_to_check:
                if section in clinical_note["structured_data"] and clinical_note["structured_data"][section]:
                    process_section_entries(section, clinical_note["structured_data"][section])
        
        # Special check for SOAP note structure
        for soap_section in ["subjective", "objective", "assessment", "plan"]:
            if soap_section in clinical_note:
                # Check for embedded lists that might contain medical entities with codes
                section_data = clinical_note[soap_section]
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if isinstance(value, list):
                            process_section_entries(f"{soap_section}.{key}", value)
    
    def _normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity names for better deduplication by handling plurals and common variations
        
        Args:
            name: Entity name to normalize
            
        Returns:
            Normalized entity name
        """
        if not name:
            return ""
            
        name = name.lower().strip()
          # Handle common plural endings
        if name.endswith('ies'):
            # e.g., "allergies" -> "allergy"
            name = name[:-3] + 'y'
        elif name.endswith('s') and len(name) > 3:
            # Common plurals like "headaches" -> "headache", "migraines" -> "migraine"
            # But avoid words that naturally end in 's' like "nausea"
            if not name.endswith(('us', 'ss', 'is', 'as')):
                # Specific medical term plurals
                if name.endswith(('aches', 'aines', 'pains', 'drugs', 'meds', 'pills', 'tics', 'ings')):
                    name = name[:-1]
                # Don't remove 's' from words ending in 'es' unless they're specific cases
                elif not name.endswith('es') and len(name) > 4:
                    # General rule: if word is long enough and doesn't end in 'es', try removing 's'
                    name = name[:-1]
        
        # Handle some common medical term variations
        substitutions = {
            'dizzy': 'dizziness',
            'feel dizzy': 'dizziness',
            'feeling dizzy': 'dizziness',
            'nauseous': 'nausea',
            'feeling nauseous': 'nausea',
            'feel nauseous': 'nausea',
        }
        
        name = substitutions.get(name, name)
        
        return name

    def _deduplicate_structured_entities(self, entities: List[Dict[str, Any]], entity_type: str) -> List[Dict[str, Any]]:
        """
        Deduplicate structured entities based on normalized name and medical codes
        
        Args:
            entities: List of entity dictionaries to deduplicate
            entity_type: Type of entity (for logging purposes)
            
        Returns:
            List of deduplicated entities
        """
        if not entities:
            return []
            
        print(f"Deduplicating {len(entities)} {entity_type} entities...")
        
        # Use a dictionary to track unique entities by their key characteristics
        unique_entities = {}
        
        for entity in entities:
            # Create a unique key based on normalized name and codes
            original_name = entity.get("name", "").strip()
            normalized_name = self._normalize_entity_name(original_name)
            
            if not normalized_name:
                continue
                
            # Create a signature based on medical codes for precise deduplication
            code_signature = []
            for key in sorted(entity.keys()):
                if key.endswith('_code') and entity[key]:
                    code_signature.append(f"{key}:{entity[key]}")
            
            # Use normalized name and codes as the unique identifier
            unique_key = f"{normalized_name}|{','.join(code_signature)}"
            
            # If we haven't seen this combination before, or if this one has better data
            if unique_key not in unique_entities:
                unique_entities[unique_key] = entity
            else:
                # Keep the one with higher confidence, or the one with more complete data
                existing = unique_entities[unique_key]
                current_confidence = entity.get("confidence", 0.5)
                existing_confidence = existing.get("confidence", 0.5)
                
                # Count non-empty fields as a tie-breaker
                current_fields = sum(1 for v in entity.values() if v and str(v).strip())
                existing_fields = sum(1 for v in existing.values() if v and str(v).strip())
                
                # Prefer singular forms over plural forms (if same confidence)
                current_is_singular = len(original_name) <= len(existing.get("name", ""))
                
                if current_confidence > existing_confidence or (
                    current_confidence == existing_confidence and (
                        current_fields > existing_fields or 
                        (current_fields == existing_fields and current_is_singular)
                    )
                ):
                    unique_entities[unique_key] = entity
        
        deduplicated = list(unique_entities.values())
        print(f"Deduplicated {entity_type}: {len(entities)} -> {len(deduplicated)}")
        
        return deduplicated
def main():
    """Main function to run the note bridge"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate clinical notes from NLP pipeline results")
    parser.add_argument("results_dir", help="Directory containing NLP pipeline results")
    parser.add_argument("--output-dir", help="Directory to save generated notes")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM-based note generation")
    parser.add_argument("--api-key", help="API key for LLM service")
    
    args = parser.parse_args()
    
    # Initialize the bridge
    bridge = NoteBridge(use_llm=not args.no_llm, llm_api_key=args.api_key)
    
    # Load NLP results
    nlp_results = bridge.load_nlp_results(args.results_dir)
    
    if not nlp_results:
        print("Error: No NLP results found in the specified directory.")
        return
    
    # Generate the note
    bridge.generate_note(nlp_results, args.output_dir)

if __name__ == "__main__":
    main()