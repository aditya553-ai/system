import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback
import re
# Import our modules
from template_filling import TemplateFiller
from safety_alerts import SafetyAlertGenerator
from note_renderer import NoteRenderer
from conversation_analyzer import ConversationAnalyzer
try:
    from llm_note_generator import LlmNoteGenerator
    LLM_AVAILABLE = True
except ImportError:
    print("Warning: LlmNoteGenerator not available. Only template-based generation will be used.")
    LLM_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

def generate_clinical_note(
    input_path: str, 
    output_dir: Optional[str] = None, 
    template_path: Optional[str] = None,
    generate_pdf: bool = True,
    use_llm: bool = True,
    llm_api_key: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate a clinical note from a transcript
    
    Args:
        input_path: Path to the transcript JSON file
        output_dir: Directory to save the output files (optional)
        template_path: Path to a custom template file (optional)
        generate_pdf: Whether to generate a PDF version (default: True)
        use_llm: Whether to use LLM for note generation (default: True)
        llm_api_key: API key for LLM service (optional)
        
    Returns:
        Dictionary with paths to generated files
    """
    try:
        print(f"Generating clinical note from: {input_path}")
        
        # Load input data
        with open(input_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        print(f"Successfully loaded transcript with {len(transcript_data.get('speaker_turns', []))} speaker turns")
        
        # Set default output directory if not provided
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(input_path), "notes")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base name from input file
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Enhance the transcript data with additional structured information
        print("Enhancing transcript data with context awareness...")
        enriched_transcript = enhance_transcript_data(transcript_data)
        
        # Generate the clinical note
        clinical_note = {}
        
        if use_llm:
            try:
                print("Generating clinical note using LLM...")
                # Make sure to import the LlmNoteGenerator class
                from llm_note_generator import LlmNoteGenerator
                llm_generator = LlmNoteGenerator(api_key=llm_api_key)
                clinical_note = llm_generator.generate_soap_note(enriched_transcript)
                print("Successfully generated SOAP note with LLM")
            except Exception as e:
                print(f"Error using LLM for note generation: {e}")
                print("Falling back to template-based note generation")
                use_llm = False
        
        if not use_llm:
            print("Using template-based note generation...")
            # Initialize template filler
            template_filler = TemplateFiller(template_path)
            clinical_note = template_filler.generate_note(enriched_transcript, "soap")
        
        # Generate safety alerts
        print("Generating safety alerts...")
        safety_alert_gen = SafetyAlertGenerator()
        try:
            safety_alerts = safety_alert_gen.generate_alerts(clinical_note)
            
            # Add alerts to the clinical note
            if safety_alerts:
                clinical_note["alerts"] = safety_alerts
                print(f"Added {len(safety_alerts)} safety alerts to the note")
            else:
                clinical_note["alerts"] = []
                print("No safety alerts generated")
        except Exception as e:
            print(f"Error generating safety alerts: {e}")
            clinical_note["alerts"] = []
        
        # Add date to the clinical note if not present
        if "date" not in clinical_note:
            clinical_note["date"] = datetime.now().strftime("%Y-%m-%d")
        
        # Output paths
        output_paths = {}
        
        # Save the clinical note as JSON for debugging
        json_path = os.path.join(output_dir, f"{base_name}_clinical_note.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clinical_note, f, indent=2)
        output_paths["json"] = json_path
        print(f"Saved clinical note JSON to: {json_path}")
        
        # Initialize the renderer
        note_renderer = NoteRenderer()
        
        # Create text version
        try:
            text_content = note_renderer.render_text(clinical_note)
            text_path = os.path.join(output_dir, f"{base_name}_clinical_note.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            output_paths["text"] = text_path
            print(f"Saved text note to: {text_path}")
        except Exception as e:
            print(f"Error generating text version: {e}")
        
        # Create HTML version
        try:
            html_content = note_renderer.render_html(clinical_note)
            html_path = os.path.join(output_dir, f"{base_name}_clinical_note.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            output_paths["html"] = html_path
            print(f"Saved HTML note to: {html_path}")
        except Exception as e:
            print(f"Error generating HTML version: {e}")
        
        # Create PDF version if requested
        if generate_pdf:
            try:
                import pdfkit
                pdf_path = os.path.join(output_dir, f"{base_name}_clinical_note.pdf")
                try:
                    if hasattr(note_renderer, 'render_pdf'):
                        pdf_result = note_renderer.render_pdf(clinical_note, pdf_path)
                        if pdf_result:
                            output_paths["pdf"] = pdf_path
                            print(f"Saved PDF note to: {pdf_path}")
                    else:
                        print("Warning: render_pdf method not found in NoteRenderer class")
                except Exception as e:
                    print(f"Error generating PDF: {e}")
                    traceback.print_exc()
            except ImportError:
                print("To enable PDF generation, install either:")
                print("1. wkhtmltopdf (https://wkhtmltopdf.org/downloads.html) and pdfkit (pip install pdfkit)")
                print("2. WeasyPrint (pip install weasyprint) and its dependencies")
        
        print("Note generation completed successfully!")
        print("Generated files:")
        for fmt, path in output_paths.items():
            print(f"- {fmt.upper()}: {path}")
        
        return output_paths
    except Exception as e:
        print(f"Error generating clinical note: {e}")
        traceback.print_exc()
        print("Note generation failed. See errors above.")
        return {}

def check_pdf_dependencies():
    """Check for PDF generation dependencies and install if missing"""
    try:
        import pdfkit
        print("Found pdfkit for PDF generation")
    except ImportError:
        print("pdfkit not found, attempting to install...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfkit"])
            print("Successfully installed pdfkit")
        except Exception as e:
            print(f"Failed to install pdfkit: {e}")
            print("You will need to install it manually: pip install pdfkit")
            print("You also need to install wkhtmltopdf from: https://wkhtmltopdf.org/downloads.html")
            print("Note generation will continue but PDF output may not be available")


def enhance_transcript_data(transcript_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance the transcript data to improve note generation
    by adding more detailed information extracted from the text
    """
    enhanced = transcript_data.copy()
    
    # Extract basic information about speakers
    patient_speaker = _identify_patient_speaker(transcript_data)
    doctor_speaker = _identify_doctor_speaker(transcript_data)
    
    # Extract structured information
    symptoms = _extract_basic_symptoms(transcript_data, patient_speaker)
    medications = _extract_basic_medications(transcript_data)
    diagnoses = _extract_basic_diagnoses(transcript_data, doctor_speaker)
    chief_complaint = _extract_basic_chief_complaint(transcript_data, symptoms, patient_speaker)
    
    # Add structured enhancements to the transcript
    if "structured_data" not in enhanced:
        enhanced["structured_data"] = {}
    
    enhanced["structured_data"]["symptoms"] = symptoms
    enhanced["structured_data"]["diagnoses"] = diagnoses
    enhanced["structured_data"]["medications"] = medications
    enhanced["structured_data"]["chief_complaint"] = chief_complaint
    enhanced["structured_data"]["history_of_present_illness"] = _extract_basic_hpi(transcript_data, patient_speaker)
    enhanced["structured_data"]["patient_speaker"] = patient_speaker
    enhanced["structured_data"]["doctor_speaker"] = doctor_speaker
    
    return enhanced

def _identify_patient_speaker(transcript_data: Dict[str, Any]) -> Optional[int]:
    """Identify which speaker is the patient"""
    # Simple heuristic - look for speakers talking about their symptoms
    patient_phrases = ["i feel", "i am", "i'm", "i have", "my pain", "my symptoms"]
    
    speaker_scores = {}
    for turn in transcript_data.get("speaker_turns", []):
        speaker = turn.get("speaker")
        text = turn.get("text", "").lower()
        
        if speaker not in speaker_scores:
            speaker_scores[speaker] = 0
        
        for phrase in patient_phrases:
            if phrase in text:
                speaker_scores[speaker] += 1
    
    if speaker_scores:
        return max(speaker_scores.items(), key=lambda x: x[1])[0]
    return 0  # Default to first speaker if we can't determine

def _identify_doctor_speaker(transcript_data: Dict[str, Any]) -> Optional[int]:
    """Identify which speaker is the doctor"""
    # Simple heuristic - look for speakers giving medical advice
    doctor_phrases = ["i recommend", "i'll prescribe", "your symptoms", "we'll need", 
                     "your condition", "the treatment"]
    
    speaker_scores = {}
    for turn in transcript_data.get("speaker_turns", []):
        speaker = turn.get("speaker")
        text = turn.get("text", "").lower()
        
        if speaker not in speaker_scores:
            speaker_scores[speaker] = 0
        
        for phrase in doctor_phrases:
            if phrase in text:
                speaker_scores[speaker] += 1
    
    if speaker_scores:
        return max(speaker_scores.items(), key=lambda x: x[1])[0]
    return 1  # Default to second speaker if we can't determine

def _extract_basic_symptoms(transcript_data: Dict[str, Any], patient_speaker: Optional[int]) -> List[Dict[str, Any]]:
    """Extract basic symptom information"""
    symptoms = []
    
    # First check normalized entities
    if "normalized_entities" in transcript_data:
        for entity_text, entity_data in transcript_data["normalized_entities"].items():
            if entity_data.get("type", "").lower() == "symptom":
                symptoms.append({
                    "name": entity_text,
                    "description": f"Patient reports {entity_text}"
                })
    
    # Check entities in turns if needed
    if not symptoms:
        for turn in transcript_data.get("speaker_turns", []):
            # Focus on patient statements if we know who the patient is
            if patient_speaker is not None and turn.get("speaker") != patient_speaker:
                continue
                
            for entity in turn.get("entities", []):
                entity_type = entity.get("type", "").lower()
                if not entity_type and "label" in entity:
                    entity_type = entity["label"].lower()
                
                if entity_type == "symptom":
                    symptoms.append({
                        "name": entity.get("text", ""),
                        "description": f"Patient reports {entity.get('text', '')}"
                    })
    
    return symptoms

def _extract_basic_medications(transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract basic medication information"""
    medications = []
    
    # Check normalized entities first
    if "normalized_entities" in transcript_data:
        for entity_text, entity_data in transcript_data["normalized_entities"].items():
            if entity_data.get("type", "").lower() in ["medication", "drug"]:
                medications.append({
                    "name": entity_text,
                    "dose": "as prescribed",
                    "frequency": "as directed"
                })
    
    # Check entities in turns if needed
    if not medications:
        for turn in transcript_data.get("speaker_turns", []):
            for entity in turn.get("entities", []):
                entity_type = entity.get("type", "").lower()
                if not entity_type and "label" in entity:
                    entity_type = entity["label"].lower()
                
                if entity_type in ["medication", "drug"]:
                    medications.append({
                        "name": entity.get("text", ""),
                        "dose": "as prescribed",
                        "frequency": "as directed"
                    })
    
    return medications

def _extract_basic_diagnoses(transcript_data: Dict[str, Any], doctor_speaker: Optional[int]) -> List[Dict[str, Any]]:
    """Extract basic diagnosis information"""
    diagnoses = []
    
    # Check normalized entities first
    if "normalized_entities" in transcript_data:
        for entity_text, entity_data in transcript_data["normalized_entities"].items():
            if entity_data.get("type", "").lower() in ["condition", "disease", "diagnosis"]:
                diagnoses.append({
                    "name": entity_text
                })
    
    # Check entities in turns if needed
    if not diagnoses:
        for turn in transcript_data.get("speaker_turns", []):
            # Focus on doctor statements if we know who the doctor is
            if doctor_speaker is not None and turn.get("speaker") != doctor_speaker:
                continue
                
            for entity in turn.get("entities", []):
                entity_type = entity.get("type", "").lower()
                if not entity_type and "label" in entity:
                    entity_type = entity["label"].lower()
                
                if entity_type in ["condition", "disease", "diagnosis"]:
                    diagnoses.append({
                        "name": entity.get("text", "")
                    })
    
    return diagnoses

def _extract_basic_chief_complaint(transcript_data: Dict[str, Any], 
                                 symptoms: List[Dict[str, Any]],
                                 patient_speaker: Optional[int]) -> str:
    """Extract basic chief complaint"""
    
    # First try to extract from first patient turn
    if patient_speaker is not None:
        for turn in transcript_data.get("speaker_turns", []):
            if turn.get("speaker") == patient_speaker:
                text = turn.get("text", "")
                if len(text.split()) > 5:  # Reasonable length statement
                    return f"Patient presents with: {text}"
                break  # Just check the first patient turn
    
    # If that fails, use the first symptom
    if symptoms:
        symptom_names = [s.get("name") for s in symptoms if s.get("name")]
        if symptom_names:
            return f"Patient presents with {symptom_names[0]}"
    
    # Generic fallback
    return "Patient presents for evaluation"

def _extract_basic_hpi(transcript_data: Dict[str, Any], patient_speaker: Optional[int]) -> str:
    """Extract basic history of present illness"""
    patient_statements = []
    
    # Get the first few substantive patient statements
    if patient_speaker is not None:
        for turn in transcript_data.get("speaker_turns", []):
            if turn.get("speaker") == patient_speaker:
                text = turn.get("text", "")
                if len(text.split()) > 5:  # Reasonable length statement
                    patient_statements.append(text)
                    if len(patient_statements) >= 2:  # Get first two substantive statements
                        break
    
    # If we found patient statements, use them
    if patient_statements:
        return " ".join(patient_statements)
    
    # Generic fallback
    return "Patient reports onset of symptoms recently. Patient notes the symptoms have been bothersome."

def main():
    """Main function for the script"""
    # Check for PDF dependencies
    check_pdf_dependencies()
    
    parser = argparse.ArgumentParser(description="Generate a clinical note from a transcript")
    parser.add_argument("input_path", help="Path to the transcript JSON file")
    parser.add_argument("--output-dir", help="Directory to save the output files")
    parser.add_argument("--template", help="Path to a custom template file")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM generation (use templates only)")
    parser.add_argument("--llm-api-key", help="API key for LLM service")
    
    args = parser.parse_args()
    
    # If LLM is not available, force use_llm to False
    use_llm_flag = not args.no_llm and LLM_AVAILABLE
    
    # Generate the clinical note with proper arguments
    result = generate_clinical_note(
        input_path=args.input_path,
        output_dir=args.output_dir,
        template_path=args.template,
        generate_pdf=not args.no_pdf,
        use_llm=use_llm_flag,
        llm_api_key=args.llm_api_key
    )
    
    # Display results summary
    if result:
        print("Note generation completed successfully!")
    else:
        print("Note generation failed or produced no outputs.")

if __name__ == "__main__":
    main()