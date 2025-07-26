from typing import Dict, List, Any, Tuple
import re

class ConversationAnalyzer:
    """
    Analyzes the flow of a medical conversation to extract contextual information
    beyond simple entity recognition.
    """
    
    def __init__(self):
        # Define patterns for different conversation segments
        self.patterns = {
            "current_medications": [
                r"(?:what medications|what medicines|what are you (?:taking|on)|current(?:ly)? (?:taking|on)|(?:do you take|are you on) any)",
                r"(?:i('m| am) (?:taking|on)|i take)"
            ],
            "new_prescription": [
                r"(?:i('ll| will) prescribe|let('s| us) (?:start|begin)|i('m| am) (?:putting|starting) you on|i recommend)",
                r"(?:new prescription|new medication|we('ll| will) start|you should (?:start|begin) taking)"
            ],
            "medical_history": [
                r"(?:previous(?:ly)?|past|history of|have you (?:ever|previously) had|have you been diagnosed)",
                r"(?:i('ve| have) had|i was diagnosed|in the past|years ago)"
            ],
            "chief_complaint": [
                r"(?:what brings you in|reason for visit|how can i help|what's going on|what seems to be the problem)",
                r"(?:i('m| am) here|i('ve| have) been having|i('m| am) experiencing|i came in)"
            ],
            "family_history": [
                r"(?:family history|anyone in your family|relatives|parents|mother|father)",
                r"(?:runs in(?: the)? family|my (?:mother|father|parent|brother|sister))"
            ],
            "social_history": [
                r"(?:do you (?:smoke|drink|use)|smoking|alcohol|exercise|diet|work|occupation|living situation)",
                r"(?:i (?:smoke|drink|work)|i('m| am) a)"
            ],
            "physical_exam": [
                r"(?:let(?:'s|s| me) (?:examine|check|take a look)|on examination|physical(?:ly)?)",
                r"(?:tenderness|normal|abnormal|findings)"
            ],
            "diagnostic_results": [
                r"(?:labs|test results|tests show|your (?:blood|urine|imaging|scan|x-ray)|levels are)",
                r"(?:came back|results (?:show|indicate)|positive for|negative for)"
            ],
            "plan_discussion": [
                r"(?:plan|course of action|treatment|therapy|recommend|suggest|next steps|follow(?:-| )up)",
                r"(?:we('ll| will) need to|you'll need to|i want you to)"
            ]
        }
    
    def analyze_conversation(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the conversation to extract contextually relevant information.
        
        Args:
            transcript_data: The transcript data including speaker turns
            
        Returns:
            Dictionary with contextual information
        """
        result = {
            "context_segments": {},
            "entity_context": {},
            "speaker_roles": {},
            "medications": {
                "current": [],
                "new": [],
                "discussed": []
            },
            "diagnoses": {
                "confirmed": [],
                "differential": [],
                "ruled_out": []
            }
        }
        
        # Identify doctor and patient
        doctor_speaker = self._identify_doctor_speaker(transcript_data)
        patient_speaker = self._identify_patient_speaker(transcript_data)
        
        result["speaker_roles"] = {
            "doctor": doctor_speaker,
            "patient": patient_speaker
        }
        
        # Analyze conversation flow
        turns = transcript_data.get("speaker_turns", [])
        current_context = None
        context_segments = {}
        
        # First pass - identify context segments
        for i, turn in enumerate(turns):
            text = turn.get("text", "").lower()
            speaker = turn.get("speaker")
            
            # Identify which context this turn belongs to
            for context_type, patterns in self.patterns.items():
                doctor_patterns = patterns[0]
                patient_patterns = patterns[1]
                
                # Use different patterns depending on speaker
                if speaker == doctor_speaker and re.search(doctor_patterns, text):
                    current_context = context_type
                    if current_context not in context_segments:
                        context_segments[current_context] = []
                    break
                elif speaker == patient_speaker and re.search(patient_patterns, text):
                    current_context = context_type
                    if current_context not in context_segments:
                        context_segments[current_context] = []
                    break
            
            # Add turn to current context
            if current_context:
                context_segments[current_context].append(i)
        
        result["context_segments"] = context_segments
        
        # Process entities within context
        entity_context = {}
        for context_type, turn_indices in context_segments.items():
            for idx in turn_indices:
                turn = turns[idx]
                for entity in turn.get("entities", []):
                    entity_text = entity.get("text")
                    if not entity_text:
                        continue
                        
                    entity_type = entity.get("type", "").lower()
                    if not entity_type and "label" in entity:
                        entity_type = entity["label"].lower()
                    
                    key = f"{entity_type}:{entity_text}"
                    if key not in entity_context:
                        entity_context[key] = []
                    
                    entity_context[key].append(context_type)
        
        result["entity_context"] = entity_context
        
        # Categorize medications based on context
        for key, contexts in entity_context.items():
            if key.startswith("medication:") or key.startswith("drug:"):
                entity_text = key.split(":", 1)[1]
                
                if any(c == "current_medications" for c in contexts):
                    result["medications"]["current"].append(entity_text)
                elif any(c == "new_prescription" for c in contexts):
                    result["medications"]["new"].append(entity_text)
                else:
                    result["medications"]["discussed"].append(entity_text)
        
        # Categorize diagnoses based on context and confirmation language
        diagnosis_contexts = {}
        for i, turn in enumerate(turns):
            text = turn.get("text", "").lower()
            speaker = turn.get("speaker")
            
            if speaker != doctor_speaker:
                continue
                
            # Check for entities in this turn
            for entity in turn.get("entities", []):
                entity_text = entity.get("text")
                entity_type = entity.get("type", "").lower()
                if not entity_type and "label" in entity:
                    entity_type = entity["label"].lower()
                
                if entity_type not in ["condition", "diagnosis", "disease"]:
                    continue
                
                # Check confirmation language
                if re.search(r"(?:confirmed|diagnosed with|has|suffering from|presenting with)", text):
                    result["diagnoses"]["confirmed"].append(entity_text)
                elif re.search(r"(?:possible|could be|might be|consider|differential|suspected)", text):
                    result["diagnoses"]["differential"].append(entity_text)
                elif re.search(r"(?:ruled out|not|doesn't have|negative for)", text):
                    result["diagnoses"]["ruled_out"].append(entity_text)
                else:
                    result["diagnoses"]["differential"].append(entity_text)
        
        # Remove duplicates
        for category in result["medications"]:
            result["medications"][category] = list(set(result["medications"][category]))
        for category in result["diagnoses"]:
            result["diagnoses"][category] = list(set(result["diagnoses"][category]))
        
        return result
    
    def _identify_doctor_speaker(self, transcript_data: Dict[str, Any]) -> int:
        """Identify which speaker is the doctor"""
        doctor_indicators = [
            "how are you feeling", "tell me about your", "what symptoms", 
            "have you been", "have you experienced", "your symptoms", 
            "i recommend", "i suggest", "i'm going to prescribe", 
            "let's discuss", "we'll need to", "you should take"
        ]
        
        speaker_scores = {}
        for turn in transcript_data.get("speaker_turns", []):
            speaker = turn.get("speaker")
            text = turn.get("text", "").lower()
            
            if speaker not in speaker_scores:
                speaker_scores[speaker] = 0
                
            for indicator in doctor_indicators:
                if indicator in text:
                    speaker_scores[speaker] += 1
            
            # Extra points for prescribing language
            if re.search(r"(?:prescribe|recommend|dosage|prescription|treatment plan)", text):
                speaker_scores[speaker] += 2
        
        if speaker_scores:
            return max(speaker_scores.items(), key=lambda x: x[1])[0]
        return None
    
    def _identify_patient_speaker(self, transcript_data: Dict[str, Any]) -> int:
        """Identify which speaker is the patient"""
        patient_indicators = [
            "i've been", "i have been", "i am", "i'm", "i feel", "i felt", 
            "my pain", "my symptom", "my health", "my condition",
            "i experienced", "i've experienced", "i've had", "i have had"
        ]
        
        speaker_scores = {}
        for turn in transcript_data.get("speaker_turns", []):
            speaker = turn.get("speaker")
            text = turn.get("text", "").lower()
            
            if speaker not in speaker_scores:
                speaker_scores[speaker] = 0
                
            for indicator in patient_indicators:
                if indicator in text:
                    speaker_scores[speaker] += 1
            
            # Look for symptom descriptions
            if re.search(r"(?:pain|discomfort|feeling|felt|experienced|having|symptom)", text):
                speaker_scores[speaker] += 1
        
        if speaker_scores:
            return max(speaker_scores.items(), key=lambda x: x[1])[0]
        return None