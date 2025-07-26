import os
import sys
import json
import requests
from datetime import datetime
import argparse

# Add path to find the medical pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "medical-nlp-pipeline", "src"))
from pipeline import MedicalNLPPipeline

class MedicalTranscriptionToPrescriptionBridge:
    """Bridge between medical transcription pipeline and auto-prescription API"""
    
    def __init__(self, prescription_api_url="http://localhost:8000"):
        """Initialize the bridge with the prescription API URL"""
        self.prescription_api_url = prescription_api_url
        self.pipeline = MedicalNLPPipeline()
        print(f"Initialized bridge with prescription API at: {prescription_api_url}")
    
    def process_transcript(self, transcript_path):
        """Process a transcript file through the medical NLP pipeline"""
        print(f"Processing transcript: {transcript_path}")
        
        # Run the medical NLP pipeline
        result = self.pipeline.run(transcript_path)
        
        if not result:
            print("Failed to process transcript")
            return None
            
        return result
    
    def extract_patient_data(self, soap_note_path):
        """Extract patient data from a SOAP note JSON file"""
        print(f"Extracting patient data from: {soap_note_path}")
        
        try:
            with open(soap_note_path, 'r', encoding='utf-8') as f:
                soap_note = json.load(f)
                
            # Extract symptoms from the SOAP note
            symptoms = []
            chronic_conditions = []
            
            # Get demographics
            age = soap_note.get("age", 50)  # Default to 50 if not found
            gender = soap_note.get("gender", "Unknown")
            patient_id = soap_note.get("patient_id", "P001")
            
            # Extract symptoms from structured data
            if "structured_data" in soap_note:
                structured = soap_note["structured_data"]
                
                # Extract symptoms
                if "symptoms" in structured:
                    for symptom in structured["symptoms"]:
                        symptoms.append(symptom["name"])
                
                # Extract chronic conditions (from problems if available)
                if "problems" in structured:
                    for problem in structured["problems"]:
                        chronic_conditions.append(problem["name"])
                        
                # Try to extract from chief complaint if no symptoms found
                if not symptoms and "chief_complaint" in structured:
                    chief_complaint = structured["chief_complaint"]
                    if "presents with" in chief_complaint:
                        # Extract the primary symptom from the chief complaint
                        primary_symptom = chief_complaint.split("presents with ")[1].split(".")[0]
                        symptoms.append(primary_symptom)
            
            # If we didn't find symptoms in structured data, try to extract from the plan
            if not symptoms and "structured_data" in soap_note and "plan" in soap_note["structured_data"]:
                plan_text = soap_note["structured_data"]["plan"]
                
                # Look for symptom terms in the plan
                symptom_terms = ["headache", "migraine", "pain", "nausea", "dizziness", 
                                "fatigue", "insomnia", "cough", "fever"]
                
                for term in symptom_terms:
                    if term in plan_text.lower():
                        symptoms.append(term)
                        break
            
            # If still no symptoms, use a default
            if not symptoms:
                symptoms = ["headache"]  # Default symptom
            
            return {
                "patient_id": patient_id,
                "age": age,
                "gender": gender,
                "symptoms": symptoms,
                "chronic_conditions": chronic_conditions
            }
                
        except Exception as e:
            print(f"Error extracting patient data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_prescription_suggestions(self, patient_data):
        """Get prescription suggestions from the auto-prescription API"""
        print(f"Requesting prescription suggestions for patient data: {patient_data}")
        
        try:
            # Make API request
            response = requests.post(
                f"{self.prescription_api_url}/prescription/suggest",
                json=patient_data,
                headers={"Content-Type": "application/json"}
            )
            
            # Check response
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error getting prescription suggestions: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_full_pipeline(self, transcript_path, output_dir=None):
        """Run the full pipeline from transcript to prescription suggestions"""
        print(f"Running full pipeline for transcript: {transcript_path}")
        
        # Set default output directory if not provided
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(transcript_path), "results")
            os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Process the transcript through the medical NLP pipeline
        result = self.process_transcript(transcript_path)
        
        if not result or "notes" not in result:
            print("Failed to generate clinical notes from transcript")
            return None
        
        # Step 2: Extract patient data from the SOAP note
        soap_note_path = result["notes"].get("json")
        if not soap_note_path:
            print("No SOAP note JSON found in results")
            return None
            
        patient_data = self.extract_patient_data(soap_note_path)
        
        if not patient_data:
            print("Failed to extract patient data from SOAP note")
            return None
            
        # Step 3: Get prescription suggestions
        prescription_suggestions = self.get_prescription_suggestions(patient_data)
        
        if not prescription_suggestions:
            print("Failed to get prescription suggestions")
            return None
            
        # Step 4: Save prescription suggestions
        suggestions_path = os.path.join(output_dir, "prescription_suggestions.json")
        with open(suggestions_path, 'w', encoding='utf-8') as f:
            json.dump(prescription_suggestions, f, indent=2)
            
        print(f"Saved prescription suggestions to: {suggestions_path}")
        
        # Print summary of suggestions
        print("\nPrescription Suggestions:")
        for medicine in prescription_suggestions.get("medicines", []):
            print(f"- {medicine['medicineName']} {medicine['dosage']} - {medicine['instructions']}")
            
        print(f"\nDoctor's Advice: {prescription_suggestions.get('doctorAdvice', '')[:100]}...")
        print(f"Follow-up Date: {prescription_suggestions.get('followUpDate', '')}")
        
        return {
            "transcript_results": result,
            "patient_data": patient_data,
            "prescription_suggestions": prescription_suggestions,
            "suggestions_path": suggestions_path
        }

def main():
    """Main function to run the bridge from command line"""
    parser = argparse.ArgumentParser(description="Process medical transcripts and get prescription suggestions")
    parser.add_argument("transcript_path", help="Path to transcript file")
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL for the auto-prescription API")
    parser.add_argument("--output-dir", help="Directory to save output files")
    
    args = parser.parse_args()
    
    bridge = MedicalTranscriptionToPrescriptionBridge(args.api_url)
    bridge.run_full_pipeline(args.transcript_path, args.output_dir)

if __name__ == "__main__":
    main()