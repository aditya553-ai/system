import os
import json
from dotenv import load_dotenv
import requests
from typing import Dict, List, Any, Optional
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlmNoteGenerator:
    """
    Class for generating clinical notes using an LLM API
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        """
        Initialize the LLM note generator
        
        Args:
            api_key: API key for the LLM service
            model: LLM model to use
        """
        # Try to get API key from environment variable if not provided
        load_dotenv()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. Set OPENAI_API_KEY environment variable or pass api_key parameter")
        
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        self.max_retries = 3
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_soap_note(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a SOAP note using an LLM based on transcript data
        
        Args:
            transcript_data: Transcript data with normalized entities
            
        Returns:
            Dictionary containing the structured SOAP note
        """
        if not self.api_key:
            raise ValueError("API key not provided")
        
        # Extract patient and doctor speakers from transcript
        patient_speaker = None
        doctor_speaker = None
        
        if "structured_data" in transcript_data:
            patient_speaker = transcript_data["structured_data"].get("patient_speaker")
            doctor_speaker = transcript_data["structured_data"].get("doctor_speaker")
        
        if patient_speaker is None or doctor_speaker is None:
            patient_speaker, doctor_speaker = self._identify_speakers(transcript_data)
        
        # Create a dialogue representation
        dialogue = self._format_dialogue(transcript_data, patient_speaker, doctor_speaker)
        
        # Extract key medical entities
        entities_info = self._extract_entities_info(transcript_data)
        
        # Create structured data summary
        structured_summary = self._create_structured_summary(transcript_data)
        
        # Prepare the prompt
        system_prompt = """You are an expert medical scribe. Your task is to create a comprehensive and accurate SOAP note based on a medical conversation.
Follow the standard SOAP format (Subjective, Objective, Assessment, Plan) with proper medical terminology.
Be thorough but concise. Include all relevant medical information while maintaining a professional clinical style.
Ensure all information in the note comes directly from the provided conversation and entities."""

        user_prompt = f"""Generate a detailed SOAP format clinical note based on this patient-doctor conversation.

=== CONVERSATION TRANSCRIPT ===
{dialogue}

=== MEDICAL ENTITIES DETECTED ===
{entities_info}

=== STRUCTURED DATA SUMMARY ===
{structured_summary}

Please format your response as a structured JSON object with these sections:
1. subjective: Object containing "chief_complaint", "history_of_present_illness", and optionally "current_medications"
2. objective: Object containing "vitals" and "physical_exam"
3. assessment: Object containing "diagnosis" (primary diagnosis and any differential diagnoses)
4. plan: Object containing "current_medications", "new_prescriptions" if applicable, and "plan_text" (additional plan details)

Be specific about which medications are current and which are newly prescribed.
Use appropriate medical terminology throughout.
Do not include information that isn't supported by the conversation or entity data.
Do not include placeholders or assumptions - if information is not provided, indicate that it was not assessed or documented."""

        try:
            # Call the LLM API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,  # Low temperature for more factual output
                "response_format": {"type": "json_object"}
            }
            
            logger.info(f"Calling LLM API with model: {self.model}")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            result = response.json()
            soap_note_json = result["choices"][0]["message"]["content"]
            
            # Parse the generated JSON
            try:
                soap_note = json.loads(soap_note_json)
                logger.info("Successfully generated SOAP note with LLM")
                return soap_note
            except json.JSONDecodeError:
                logger.error(f"Error parsing LLM output as JSON: {soap_note_json[:100]}...")
                return self._generate_fallback_soap_note(transcript_data)
        
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return self._generate_fallback_soap_note(transcript_data)
    
    def _identify_speakers(self, transcript_data: Dict[str, Any]) -> tuple:
        """Identify patient and doctor speakers from transcript"""
        patient_speaker = 0  # Default to first speaker
        doctor_speaker = 1   # Default to second speaker
        
        # Simple heuristics for identification
        patient_indicators = ["i feel", "i have", "my pain", "i am experiencing", "i've been"]
        doctor_indicators = ["prescribe", "recommend", "diagnosis", "treatment", "let me examine"]
        
        speaker_roles = {}
        
        for turn in transcript_data.get("speaker_turns", []):
            speaker = turn.get("speaker")
            text = turn.get("text", "").lower()
            
            if speaker not in speaker_roles:
                speaker_roles[speaker] = {"patient_score": 0, "doctor_score": 0}
            
            # Score patient indicators
            for indicator in patient_indicators:
                if indicator in text:
                    speaker_roles[speaker]["patient_score"] += 1
            
            # Score doctor indicators
            for indicator in doctor_indicators:
                if indicator in text:
                    speaker_roles[speaker]["doctor_score"] += 1
        
        # Determine roles based on scores
        if speaker_roles:
            patient_candidates = sorted(speaker_roles.items(), key=lambda x: x[1]["patient_score"], reverse=True)
            doctor_candidates = sorted(speaker_roles.items(), key=lambda x: x[1]["doctor_score"], reverse=True)
            
            if patient_candidates and patient_candidates[0][1]["patient_score"] > 0:
                patient_speaker = patient_candidates[0][0]
            
            if doctor_candidates and doctor_candidates[0][1]["doctor_score"] > 0:
                doctor_speaker = doctor_candidates[0][0]
            
            # Make sure they're different speakers
            if patient_speaker == doctor_speaker and len(speaker_roles) > 1:
                if doctor_candidates[0][1]["doctor_score"] > patient_candidates[0][1]["patient_score"]:
                    patient_speaker = [spk for spk in speaker_roles.keys() if spk != doctor_speaker][0]
                else:
                    doctor_speaker = [spk for spk in speaker_roles.keys() if spk != patient_speaker][0]
        
        return patient_speaker, doctor_speaker
    
    def _format_dialogue(self, transcript_data: Dict[str, Any], patient_speaker: int, doctor_speaker: int) -> str:
        """Format the dialogue for the LLM prompt"""
        formatted_lines = []
        
        for turn in transcript_data.get("speaker_turns", []):
            speaker = turn.get("speaker")
            text = turn.get("text", "")
            
            if speaker == patient_speaker:
                formatted_lines.append(f"Patient: {text}")
            elif speaker == doctor_speaker:
                formatted_lines.append(f"Doctor: {text}")
            else:
                formatted_lines.append(f"Speaker {speaker}: {text}")
        
        return "\n".join(formatted_lines)
    
    def _extract_entities_info(self, transcript_data: Dict[str, Any]) -> str:
        """Extract and format medical entity information"""
        entity_info_lines = []
        
        # Check for normalized entities first
        if "normalized_entities" in transcript_data:
            entity_info_lines.append("Normalized Medical Entities:")
            
            # Group entities by type
            entities_by_type = {}
            for entity_text, entity_data in transcript_data["normalized_entities"].items():
                entity_type = entity_data.get("type", "unknown").lower()
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                
                entities_by_type[entity_type].append({
                    "text": entity_text,
                    "codes": entity_data.get("codes", {})
                })
            
            # Format each entity type
            for entity_type, entities in entities_by_type.items():
                entity_info_lines.append(f"- {entity_type.capitalize()}: {', '.join(e['text'] for e in entities)}")
        
        # If no normalized entities, extract from speaker turns
        elif "speaker_turns" in transcript_data:
            entity_info_lines.append("Extracted Medical Entities:")
            all_entities = []
            
            for turn in transcript_data.get("speaker_turns", []):
                for entity in turn.get("entities", []):
                    entity_type = entity.get("type", "").lower()
                    if not entity_type and "label" in entity:
                        entity_type = entity["label"].lower()
                    
                    entity_text = entity.get("text", "")
                    if entity_text and entity_type:
                        all_entities.append((entity_type, entity_text))
            
            # Group by type
            entities_by_type = {}
            for entity_type, entity_text in all_entities:
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                
                if entity_text not in entities_by_type[entity_type]:
                    entities_by_type[entity_type].append(entity_text)
            
            # Format each entity type
            for entity_type, entity_texts in entities_by_type.items():
                entity_info_lines.append(f"- {entity_type.capitalize()}: {', '.join(entity_texts)}")
        
        return "\n".join(entity_info_lines)
    
    def _create_structured_summary(self, transcript_data: Dict[str, Any]) -> str:
        """Create a structured summary from available data"""
        summary_lines = []
        
        # Add structured data if available
        if "structured_data" in transcript_data:
            structured_data = transcript_data["structured_data"]
            
            # Chief complaint
            if "chief_complaint" in structured_data:
                summary_lines.append(f"Chief Complaint: {structured_data['chief_complaint']}")
            
            # Symptoms
            if "symptoms" in structured_data and structured_data["symptoms"]:
                symptoms_text = ", ".join([s.get("name", "") for s in structured_data["symptoms"] if s.get("name")])
                if symptoms_text:
                    summary_lines.append(f"Reported Symptoms: {symptoms_text}")
            
            # Medications
            if "current_medications" in structured_data and structured_data["current_medications"]:
                meds_text = ", ".join([m.get("name", "") for m in structured_data["current_medications"] if m.get("name")])
                if meds_text:
                    summary_lines.append(f"Current Medications: {meds_text}")
            
            if "new_medications" in structured_data and structured_data["new_medications"]:
                meds_text = ", ".join([m.get("name", "") for m in structured_data["new_medications"] if m.get("name")])
                if meds_text:
                    summary_lines.append(f"New Medications: {meds_text}")
            
            # Diagnoses
            if "diagnoses" in structured_data and structured_data["diagnoses"]:
                diag_text = ", ".join([d.get("name", "") for d in structured_data["diagnoses"] if d.get("name")])
                if diag_text:
                    summary_lines.append(f"Diagnoses: {diag_text}")
        
        return "\n".join(summary_lines)
    
    def _generate_fallback_soap_note(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a basic SOAP note when LLM generation fails"""
        logger.info("Generating fallback SOAP note")
        
        soap_note = {
            "subjective": {
                "chief_complaint": "Patient presents for evaluation",
                "history_of_present_illness": "Patient reports symptoms"
            },
            "objective": {
                "vitals": {
                    "temperature": "Not documented",
                    "heart_rate": "Not documented",
                    "blood_pressure": "Not documented",
                    "respiratory_rate": "Not documented",
                    "oxygen_saturation": "Not documented"
                },
                "physical_exam": "Physical examination was not documented"
            },
            "assessment": {
                "diagnosis": "Assessment pending further evaluation"
            },
            "plan": {
                "plan_text": "Follow up as needed for symptom management"
            }
        }
        
        # Try to enhance with any available structured data
        if "structured_data" in transcript_data:
            sd = transcript_data["structured_data"]
            
            if "chief_complaint" in sd:
                soap_note["subjective"]["chief_complaint"] = sd["chief_complaint"]
            
            if "history_of_present_illness" in sd:
                soap_note["subjective"]["history_of_present_illness"] = sd["history_of_present_illness"]
            
            if "current_medications" in sd:
                current_meds = []
                for med in sd["current_medications"]:
                    if isinstance(med, dict) and "name" in med:
                        current_meds.append(med)
                    elif isinstance(med, str):
                        current_meds.append({"name": med, "dose": "as prescribed", "frequency": "as directed"})
                
                if current_meds:
                    soap_note["plan"]["current_medications"] = current_meds
            
            if "new_medications" in sd:
                new_meds = []
                for med in sd["new_medications"]:
                    if isinstance(med, dict) and "name" in med:
                        new_meds.append(med)
                    elif isinstance(med, str):
                        new_meds.append({"name": med, "dose": "as prescribed", "frequency": "as directed"})
                
                if new_meds:
                    soap_note["plan"]["new_prescriptions"] = new_meds
            
            if "diagnoses" in sd:
                diagnoses = []
                for diag in sd["diagnoses"]:
                    if isinstance(diag, dict) and "name" in diag:
                        diagnoses.append(diag["name"])
                    elif isinstance(diag, str):
                        diagnoses.append(diag)
                
                if diagnoses:
                    soap_note["assessment"]["diagnosis"] = ", ".join(diagnoses)
        
        return soap_note