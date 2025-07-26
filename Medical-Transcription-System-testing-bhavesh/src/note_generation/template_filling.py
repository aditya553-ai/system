import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class TemplateFiller:
    """
    Class for filling clinical note templates with extracted and normalized entities
    """
    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize with a template
        
        Args:
            template_path: Path to JSON template file (or use default SOAP template)
        """
        if template_path and os.path.exists(template_path):
            with open(template_path, 'r') as f:
                self.template = json.load(f)
        else:
            # Default SOAP note template
            self.template = {
                "metadata": {
                    "title": "Clinical Note",
                    "document_type": "SOAP Note",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                },
                "patient_info": {},
                "provider_info": {},
                "subjective": {
                    "chief_complaint": "",
                    "history_of_present_illness": "",
                    "symptoms": [],
                    "allergies": [],
                    "medications": [],
                    "history": {
                        "medical_history": [],
                        "surgical_history": [],
                        "family_history": [],
                        "social_history": ""
                    }
                },
                "objective": {
                    "vitals": {},
                    "physical_exam": {}
                },
                "assessment": {
                    "diagnoses": [],
                    "differential_diagnosis": []
                },
                "plan": {
                    "medications": [],
                    "tests": [],
                    "procedures": [],
                    "followup": ""
                }
            }

    def generate_note(self, transcript_data: Dict[str, Any], template_type: str = "soap") -> Dict[str, Any]:
        """
        Generate a clinical note from transcript data
        
        Args:
            transcript_data: Dictionary containing transcript and entities
            template_type: Type of note to generate (soap, progress, basic)
            
        Returns:
            Dictionary with note sections
        """
        print("Filling template with entities...")
        
        # Check if we have normalized entities
        has_normalized = "normalized_entities" in transcript_data
        print(f"Using normalized entities: {has_normalized}")
        
        if template_type.lower() == "soap":
            note = self._generate_soap_note(transcript_data)
        elif template_type.lower() == "progress":
            note = self._generate_progress_note(transcript_data)
        else:
            note = self._generate_basic_note(transcript_data)
        
        return note

    def _generate_basic_note(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a basic clinical note from transcript data"""
        note = {}
        
        # Extract chief complaint
        chief_complaint = self._extract_chief_complaint(transcript_data)
        note["chief_complaint"] = chief_complaint
        
        # Extract history of present illness
        hpi = self._extract_hpi(transcript_data)
        note["history_of_present_illness"] = hpi
        
        # Extract assessment
        assessment = self._extract_diagnosis(transcript_data)
        note["assessment"] = assessment
        
        # Extract plan
        plan = self._extract_basic_plan(transcript_data)
        note["plan"] = plan
        
        return note

    def _generate_progress_note(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a progress note from transcript data"""
        note = {}
        
        # Subjective section
        subjective = {}
        subjective["chief_complaint"] = self._extract_chief_complaint(transcript_data)
        subjective["progress"] = self._extract_narrative(transcript_data)
        note["subjective"] = subjective
        
        # Assessment section
        note["assessment"] = self._extract_diagnosis(transcript_data)
        
        # Plan section
        plan = {}
        medications = self._get_medications(transcript_data)
        if medications:
            processed_meds = self._process_medications(medications)
            plan["medications"] = processed_meds
        
        plan["instructions"] = self._extract_basic_plan(transcript_data)
        note["plan"] = plan
        
        return note

    def _generate_plan(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate treatment plan section from transcript data"""
        plan = {}
        
        # Extract medications with context awareness
        current_meds = transcript_data.get("structured_data", {}).get("current_medications", [])
        new_meds = transcript_data.get("structured_data", {}).get("new_medications", [])
        
        # Process current medications
        if current_meds:
            current_meds_processed = self._process_medications(current_meds)
            plan["current_medications"] = current_meds_processed
        
        # Process new medications (prescriptions)
        if new_meds:
            new_meds_processed = self._process_medications(new_meds)
            plan["new_prescriptions"] = new_meds_processed
        
        # Extract basic plan text
        plan_text = self._extract_basic_plan(transcript_data)
        plan["plan_text"] = plan_text
        
        return plan

    def _generate_soap_note(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a SOAP note from transcript data"""
        soap_note = {}
        
        # Subjective section
        subjective = {}
        subjective["chief_complaint"] = self._extract_chief_complaint(transcript_data)
        subjective["history_of_present_illness"] = self._extract_hpi(transcript_data)
        
        # Add medication history from current medications
        current_meds = transcript_data.get("structured_data", {}).get("current_medications", [])
        if current_meds:
            med_names = [med.get("name", "") for med in current_meds]
            subjective["current_medications"] = ", ".join(med_names)
        
        soap_note["subjective"] = subjective
        
        # Objective section
        objective = {}
        objective["vitals"] = self._extract_vitals(transcript_data)
        objective["physical_exam"] = self._extract_physical_exam(transcript_data)
        soap_note["objective"] = objective
        
        # Assessment section
        assessment = {}
        assessment["diagnosis"] = self._extract_diagnosis(transcript_data)
        soap_note["assessment"] = assessment
        
        # Plan section
        plan = self._generate_plan(transcript_data)
        soap_note["plan"] = plan
        
        return soap_note

    def fill_template(self, transcript_data: Dict[str, Any], 
                     metadata: Optional[Dict[str, Any]] = None,
                     patient_info: Optional[Dict[str, Any]] = None,
                     provider_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fill the template with entities from the transcript
        
        Args:
            transcript_data: Normalized transcript with entities
            metadata: Optional metadata to include
            patient_info: Optional patient information
            provider_info: Optional provider information
            
        Returns:
            Filled clinical note template
        """
        # Create a deep copy of the template
        clinical_note = self.template.copy()
        
        # Fill in metadata if provided
        if metadata:
            clinical_note["metadata"].update(metadata)
        
        # Fill in patient and provider info
        if patient_info:
            clinical_note["patient_info"] = patient_info
        
        if provider_info:
            clinical_note["provider_info"] = provider_info
        
        # Use structured data if available, otherwise extract from entities
        if "structured_data" in transcript_data:
            structured_data = transcript_data["structured_data"]
            
            # Fill in chief complaint
            if "chief_complaint" in structured_data:
                clinical_note["subjective"]["chief_complaint"] = structured_data["chief_complaint"]
            
            # Fill in history of present illness
            if "history_of_present_illness" in structured_data:
                clinical_note["subjective"]["history_of_present_illness"] = structured_data["history_of_present_illness"]
            
            # Fill in symptoms
            if "symptoms" in structured_data:
                clinical_note["subjective"]["symptoms"] = structured_data["symptoms"]
            
            # Fill in diagnoses
            if "diagnoses" in structured_data:
                clinical_note["assessment"]["diagnoses"] = structured_data["diagnoses"]
            
            # Fill in medications
            if "medications" in structured_data:
                clinical_note["subjective"]["medications"] = [
                    {"name": med.get("name", "")} for med in structured_data["medications"]
                ]
                clinical_note["plan"]["medications"] = structured_data["medications"]
            
            # Fill in vitals
            if "vitals" in structured_data:
                clinical_note["objective"]["vitals"] = structured_data["vitals"]
        else:
            # Extract and organize entities from the transcript
            entities = self._collect_entities(transcript_data)
            
            # Fill in chief complaint
            chief_complaint = self._extract_chief_complaint(transcript_data)
            if chief_complaint:
                clinical_note["subjective"]["chief_complaint"] = chief_complaint
            
            # Fill in symptoms
            symptoms = entities.get("symptom", [])
            clinical_note["subjective"]["symptoms"] = symptoms
            
            # Fill in medications
            medications = entities.get("medication", [])
            clinical_note["subjective"]["medications"] = [
                {"name": med.get("text", "")} for med in medications
            ]
            clinical_note["plan"]["medications"] = self._process_medications(medications)
            
            # Fill in conditions/diagnoses
            conditions = entities.get("condition", [])
            clinical_note["assessment"]["diagnoses"] = conditions
            
            # Fill in tests
            tests = entities.get("test", [])
            clinical_note["plan"]["tests"] = tests
            
            # Fill in allergies - these might be mentioned in the transcript
            allergies = entities.get("allergy", [])
            if allergies:
                clinical_note["subjective"]["allergies"] = [
                    allergy.get("text", "") for allergy in allergies
                ]
            
            # Extract the narrative from transcript for history of present illness
            clinical_note["subjective"]["history_of_present_illness"] = self._extract_narrative(transcript_data)
        
        # Add follow-up recommendation
        clinical_note["plan"]["followup"] = "Follow up in 3 months for medication review and symptom assessment."
        
        return clinical_note
    
    def _collect_entities(self, transcript_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Collect entities from transcript and organize by entity type"""
        entities_by_type = {}
        
        # Check if we have the new normalized_entities format
        if "normalized_entities" in transcript_data:
            print("Using normalized_entities from transcript data")
            
            # Process normalized entities
            for entity_text, entity_data in transcript_data["normalized_entities"].items():
                entity_type = entity_data.get("type", "unknown").lower()
                
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                    
                processed_entity = {
                    "text": entity_text,
                    "type": entity_type,
                    "normalized": entity_data.get("normalized", entity_text),
                    "codes": entity_data.get("codes", {}),
                    "confidence": entity_data.get("confidence", 0.5)
                }
                
                entities_by_type[entity_type].append(processed_entity)
            
            return entities_by_type
        
        # Fall back to processing entities from speaker turns
        # Go through all speaker turns
        for turn in transcript_data.get("speaker_turns", []):
            for entity in turn.get("entities", []):
                entity_type = entity.get("type", "unknown").lower()
                if not entity_type and "label" in entity:
                    # Convert label like 'SYMPTOM' to lowercase 'symptom'
                    entity_type = entity["label"].lower()
                    entity["type"] = entity_type
                
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                
                # Add normalization data if available
                processed_entity = entity.copy()
                if "normalization" in entity:
                    processed_entity["codes"] = entity["normalization"].get("codes", {})
                    processed_entity["normalized"] = entity["normalization"].get("normalized", entity.get("text", ""))
                
                entities_by_type[entity_type].append(processed_entity)
        
        # Remove duplicates by creating a set of texts
        for entity_type in entities_by_type:
            seen_texts = set()
            unique_entities = []
            
            for entity in entities_by_type[entity_type]:
                text = entity.get("text", "").lower()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    unique_entities.append(entity)
            
            entities_by_type[entity_type] = unique_entities
        
        return entities_by_type
    
    def _extract_diagnosis(self, transcript_data: Dict[str, Any]) -> str:
        """Extract diagnosis from transcript"""
        diagnoses = []
        
        # Try to get from structured data first
        if "structured_data" in transcript_data and "diagnoses" in transcript_data["structured_data"]:
            for diagnosis in transcript_data["structured_data"]["diagnoses"]:
                if isinstance(diagnosis, dict) and "name" in diagnosis:
                    if diagnosis.get("status") == "confirmed":
                        diagnoses.append(diagnosis["name"])
                    elif diagnosis.get("status") == "differential":
                        diagnoses.append(f"Possible {diagnosis['name']}")
                    else:
                        diagnoses.append(diagnosis["name"])
                elif isinstance(diagnosis, str):
                    diagnoses.append(diagnosis)
            
            if diagnoses:
                return ", ".join(diagnoses)
        
        # Try to extract from normalized entities
        if "normalized_entities" in transcript_data:
            for entity_text, entity_data in transcript_data["normalized_entities"].items():
                entity_type = entity_data.get("type", "").lower()
                if entity_type in ["condition", "disease", "diagnosis"]:
                    diagnoses.append(entity_text)
        
        # Try to extract from speaker turns
        if not diagnoses:
            # First identify doctor speaker
            doctor_speaker = self._identify_doctor_speaker(transcript_data)
            
            for turn in transcript_data.get("speaker_turns", []):
                # If we know doctor speaker, only check doctor turns
                if doctor_speaker is not None and turn.get("speaker") != doctor_speaker:
                    continue
                    
                # Check entities in turn
                for entity in turn.get("entities", []):
                    entity_type = entity.get("type", "").lower()
                    if not entity_type and "label" in entity:
                        entity_type = entity["label"].lower()
                        
                    if entity_type in ["condition", "disease", "diagnosis"]:
                        diagnoses.append(entity.get("text", ""))
        
        # Return formatted diagnoses
        if diagnoses:
            return ", ".join(diagnoses)
        
        # Default if no diagnosis found
        return "Assessment pending further evaluation"

    def _extract_chief_complaint(self, transcript_data: Dict[str, Any]) -> str:
        """Extract the chief complaint from transcript"""
        
        # First try to get from structured data
        if "structured_data" in transcript_data and "chief_complaint" in transcript_data["structured_data"]:
            return transcript_data["structured_data"]["chief_complaint"]
        
        # Try to identify patient
        patient_speaker = self._identify_patient_speaker(transcript_data)
        
        # Extract from first patient turn with substantial content
        for turn in transcript_data.get("speaker_turns", []):
            if patient_speaker is not None and turn.get("speaker") != patient_speaker:
                continue
                
            if len(turn.get("text", "").split()) > 5:
                # Check for symptoms in this turn
                for entity in turn.get("entities", []):
                    if entity.get("type", "").lower() == "symptom" or entity.get("label", "") == "SYMPTOM":
                        return f"Patient presents with {entity.get('text', '')}"
                
                # If no symptom found but it's a substantial patient response, use it
                return f"Patient presents with: {turn.get('text', '')[:60]}..."
        
        # If we have normalized entities, check there
        if "normalized_entities" in transcript_data:
            for entity_text, entity_data in transcript_data["normalized_entities"].items():
                if entity_data.get("type") == "symptom":
                    return f"Patient presents with {entity_text}"
        
        # If all else fails, use a generic chief complaint
        return "Patient presents for evaluation"

    def _get_symptoms(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get symptoms from transcript data"""
        
        symptoms = []
        
        # Check structured data first
        if "structured_data" in transcript_data and "symptoms" in transcript_data["structured_data"]:
            return transcript_data["structured_data"]["symptoms"]
        
        # Extract from normalized entities
        if "normalized_entities" in transcript_data:
            for entity_text, entity_data in transcript_data["normalized_entities"].items():
                entity_type = entity_data.get("type", "").lower()
                if entity_type == "symptom":
                    symptoms.append({
                        "name": entity_text,
                        "description": f"Patient reports {entity_text}",
                        "codes": entity_data.get("codes", {})
                    })
        
        # Extract from speaker turns if needed
        if not symptoms:
            patient_speaker = self._identify_patient_speaker(transcript_data)
            
            for turn in transcript_data.get("speaker_turns", []):
                # Only look in patient turns if we know who the patient is
                if patient_speaker is not None and turn.get("speaker") != patient_speaker:
                    continue
                    
                for entity in turn.get("entities", []):
                    entity_type = entity.get("type", "").lower()
                    if not entity_type and "label" in entity:
                        entity_type = entity["label"].lower()
                        
                    if entity_type == "symptom":
                        symptoms.append({
                            "name": entity.get("text", ""),
                            "description": f"Patient reports {entity.get('text', '')}",
                            "codes": entity.get("normalization", {}).get("codes", {})
                        })
        
        return symptoms

    def _generate_soap_note(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a SOAP note from transcript data"""
        soap_note = {}
        
        # Subjective section
        subjective = {}
        subjective["chief_complaint"] = self._extract_chief_complaint(transcript_data)
        subjective["history_of_present_illness"] = self._extract_hpi(transcript_data)
        
        # Add medication history from current medications
        medications = self._get_medications(transcript_data)
        if medications:
            current_meds = []
            for med in medications:
                if isinstance(med, dict) and "name" in med:
                    current_meds.append(med["name"])
                elif isinstance(med, str):
                    current_meds.append(med)
            
            if current_meds:
                subjective["current_medications"] = ", ".join(current_meds)
        
        soap_note["subjective"] = subjective
        
        # Objective section
        objective = {}
        objective["vitals"] = self._extract_vitals(transcript_data)
        objective["physical_exam"] = self._extract_physical_exam(transcript_data)
        soap_note["objective"] = objective
        
        # Assessment section
        assessment = {}
        assessment["diagnosis"] = self._extract_diagnosis(transcript_data)
        soap_note["assessment"] = assessment
        
        # Plan section
        plan = self._generate_plan(transcript_data)
        soap_note["plan"] = plan
        
        return soap_note

    def _generate_plan(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate treatment plan section from transcript data"""
        plan = {}
        
        # Extract medications with context awareness
        medications = self._get_medications(transcript_data)
        
        # Process current medications and new medications if context available
        current_meds = []
        new_meds = []
        
        # Check if we have context-separated medications
        if "structured_data" in transcript_data:
            if "current_medications" in transcript_data["structured_data"]:
                current_meds = transcript_data["structured_data"]["current_medications"]
            
            if "new_medications" in transcript_data["structured_data"]:
                new_meds = transcript_data["structured_data"]["new_medications"]
        
        # If we don't have context separation, treat all medications as current
        if not current_meds and not new_meds and medications:
            current_meds = medications
        
        # Process current medications
        if current_meds:
            current_meds_processed = self._process_medications(current_meds)
            plan["current_medications"] = current_meds_processed
        
        # Process new medications (prescriptions)
        if new_meds:
            new_meds_processed = self._process_medications(new_meds)
            plan["new_prescriptions"] = new_meds_processed
        
        # Extract basic plan text
        plan_text = self._extract_basic_plan(transcript_data)
        plan["plan_text"] = plan_text
        
        return plan
        
    def _extract_hpi(self, transcript_data: Dict[str, Any]) -> str:
        """Extract history of present illness from transcript"""
        
        # Try to get from structured data first
        if (transcript_data.get("structured_data", {}).get("history_of_present_illness")):
            return transcript_data["structured_data"]["history_of_present_illness"]
        
        # Otherwise extract from narrative
        return self._extract_narrative(transcript_data)

    def _extract_narrative(self, transcript_data: Dict[str, Any]) -> str:
        """Extract patient narrative from transcript"""
        narrative_parts = []
        
        # Try to identify the patient speaker
        patient_speaker = self._identify_patient_speaker(transcript_data)
        
        # Extract patient's statements 
        for turn in transcript_data.get("speaker_turns", []):
            # If we know the patient speaker, only include their turns
            if patient_speaker is not None:
                if turn.get("speaker") == patient_speaker and len(turn.get("text", "").split()) > 5:
                    narrative_parts.append(turn.get("text", ""))
            # Otherwise include substantive turns that don't sound like the doctor
            elif len(turn.get("text", "").split()) > 5:
                text = turn.get("text", "").lower()
                doctor_phrases = ["let me", "i recommend", "i suggest", "i'll prescribe", "we'll need"]
                is_doctor = any(phrase in text for phrase in doctor_phrases)
                if not is_doctor:
                    narrative_parts.append(turn.get("text", ""))
        
        # Combine the parts into a coherent narrative
        if narrative_parts:
            return " ".join(narrative_parts[:3])  # Limit to first 3 substantial responses
        
        # If no narrative could be extracted, provide a generic default
        return "Patient reports symptoms that prompted today's visit. Patient states the symptoms have been ongoing for several days."

    def _identify_patient_speaker(self, transcript_data: Dict[str, Any]) -> Optional[int]:
        """Try to identify which speaker is the patient"""
        # Common patient indicators in text
        patient_indicators = [
            "i've been", "i have been", "i am", "i'm", "i feel", "i felt", 
            "my pain", "my symptom", "my health", "my condition",
            "i experienced", "i've experienced", "i've had", "i have had"
        ]
        
        # Check if we already identified patient in structured data
        if "structured_data" in transcript_data and "patient_speaker" in transcript_data["structured_data"]:
            return transcript_data["structured_data"]["patient_speaker"]
        
        # Count potential patient indicators for each speaker
        speaker_scores = {}
        
        for turn in transcript_data.get("speaker_turns", []):
            speaker = turn.get("speaker")
            text = turn.get("text", "").lower()
            
            if speaker not in speaker_scores:
                speaker_scores[speaker] = 0
                
            # Check for patient indicators
            for indicator in patient_indicators:
                if indicator in text:
                    speaker_scores[speaker] += 1
                    
            # Also check for entities related to symptoms
            for entity in turn.get("entities", []):
                if entity.get("type") == "symptom" or entity.get("label") == "SYMPTOM":
                    speaker_scores[speaker] += 1
        
        # Find the speaker with the highest score
        if speaker_scores:
            max_speaker = max(speaker_scores.items(), key=lambda x: x[1])
            if max_speaker[1] > 0:  # Only if we have some confidence
                return max_speaker[0]
        
        # Couldn't identify with confidence
        return None

    def _extract_vitals(self, transcript_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract vitals from transcript"""
        
        # Check if vitals are in structured data
        if "structured_data" in transcript_data and "vitals" in transcript_data["structured_data"]:
            return transcript_data["structured_data"]["vitals"]
        
        # Default vitals if not found
        return {
            "temperature": "Within normal limits",
            "heart_rate": "Within normal limits",
            "blood_pressure": "Within normal limits",
            "respiratory_rate": "Within normal limits",
            "oxygen_saturation": "Within normal limits"
        }

    def _extract_physical_exam(self, transcript_data: Dict[str, Any]) -> str:
        """Extract physical exam findings from transcript"""
        
        # Try to get from structured data
        if "structured_data" in transcript_data and "physical_exam" in transcript_data["structured_data"]:
            return transcript_data["structured_data"]["physical_exam"]
        
        # Try to find relevant parts in the transcript
        exam_mentions = []
        physical_exam_indicators = [
            "physical exam", "examination", "on exam", "exam reveals", 
            "auscultation", "palpation", "percussion", "inspection"
        ]
        
        # Identify doctor speaker
        doctor_speaker = self._identify_doctor_speaker(transcript_data)
        
        for turn in transcript_data.get("speaker_turns", []):
            # Only consider doctor's statements if we know who the doctor is
            if doctor_speaker is not None and turn.get("speaker") != doctor_speaker:
                continue
                
            text = turn.get("text", "").lower()
            
            # Check for physical exam indicators
            for indicator in physical_exam_indicators:
                if indicator in text:
                    sentences = text.split('.')
                    for sentence in sentences:
                        if indicator in sentence:
                            exam_mentions.append(sentence.strip() + ".")
        
        if exam_mentions:
            return " ".join(exam_mentions)
        
        # Default if no physical exam findings mentioned
        return "Physical examination was within normal limits."

    def _identify_doctor_speaker(self, transcript_data: Dict[str, Any]) -> Optional[int]:
        """Try to identify which speaker is the doctor"""
        # Common doctor indicators in text
        doctor_indicators = [
            "how are you feeling", "tell me about your", "what symptoms", 
            "have you been", "have you experienced", "your symptoms", 
            "i recommend", "i suggest", "i'm going to prescribe", 
            "let's discuss", "we'll need to", "you should take"
        ]
        
        # Check if we already identified doctor in structured data
        if "structured_data" in transcript_data and "doctor_speaker" in transcript_data["structured_data"]:
            return transcript_data["structured_data"]["doctor_speaker"]
        
        # Count potential doctor indicators for each speaker
        speaker_scores = {}
        
        for turn in transcript_data.get("speaker_turns", []):
            speaker = turn.get("speaker")
            text = turn.get("text", "").lower()
            
            if speaker not in speaker_scores:
                speaker_scores[speaker] = 0
                
            # Check for doctor indicators
            for indicator in doctor_indicators:
                if indicator in text:
                    speaker_scores[speaker] += 1
                    
            # Also check for explicit mentions of being a doctor
            if "doctor" in text or "physician" in text or "dr." in text:
                speaker_scores[speaker] += 2
        
        # Find the speaker with the highest score
        if speaker_scores:
            max_speaker = max(speaker_scores.items(), key=lambda x: x[1])
            if max_speaker[1] > 0:  # Only if we have some confidence
                return max_speaker[0]
        
        # Couldn't identify with confidence
        return None
    
    def _get_medications(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get medications from transcript data"""
        
        medications = []
        
        # Check for different medication structures in the data
        if "structured_data" in transcript_data:
            structured_data = transcript_data["structured_data"]
            
            # Check current_medications first
            if "current_medications" in structured_data:
                medications.extend(structured_data["current_medications"])
            
            # Then check new_medications 
            if "new_medications" in structured_data:
                medications.extend(structured_data["new_medications"])
            
            # Fall back to generic medications field
            elif "medications" in structured_data:
                medications.extend(structured_data["medications"])
        
        # If we still don't have medications, try to find them in normalized entities
        if not medications and "normalized_entities" in transcript_data:
            for entity_text, entity_data in transcript_data["normalized_entities"].items():
                if entity_data.get("type", "").lower() in ["medication", "drug"]:
                    medications.append({
                        "name": entity_text,
                        "codes": entity_data.get("codes", {})
                    })
        
        return medications

    def _extract_basic_plan(self, transcript_data: Dict[str, Any]) -> str:
        """Extract basic plan from transcript"""
        plan_items = []
        
        # Add medications to plan
        medications = self._get_medications(transcript_data)
        if medications:
            med_strings = []
            for med in medications:
                med_string = f"{med.get('name')} {med.get('dose', 'as prescribed')} {med.get('frequency', 'as directed')}"
                med_strings.append(med_string)
            
            if med_strings:
                plan_items.append("Medications: " + ", ".join(med_strings))
        
        # Add follow-up recommendation
        plan_items.append("Follow up as needed for symptom management.")
        
        # Generate plan based on symptoms
        symptoms = []
        if "structured_data" in transcript_data and "symptoms" in transcript_data["structured_data"]:
            symptoms = transcript_data["structured_data"]["symptoms"]
        
        if symptoms:
            symptom_names = [s.get('name', '') for s in symptoms]
            plan_items.append(f"Continue to monitor {', '.join(symptom_names)}.")
        
        # Generic recommendations
        plan_items.append("Return to clinic if symptoms worsen or fail to improve.")
        
        return "\n".join(plan_items)

    def _process_medications(self, medications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process medication entities into a structured format for the plan"""
        structured_meds = []
        
        for med in medications:
            # First try to get name from "name" field, then fall back to "text"
            med_name = med.get("name", med.get("text", ""))
            
            structured_med = {
                "name": med_name
            }
            
            # Extract dosage if present
            if "dosage" in med:
                structured_med["dose"] = med["dosage"]
            elif "dose" in med:
                structured_med["dose"] = med["dose"]
            else:
                structured_med["dose"] = "as prescribed"  # Generic default
            
            # Extract frequency if present
            if "frequency" in med:
                structured_med["frequency"] = med["frequency"]
            else:
                structured_med["frequency"] = "as directed"  # Generic default
                
            structured_meds.append(structured_med)
        
        return structured_meds