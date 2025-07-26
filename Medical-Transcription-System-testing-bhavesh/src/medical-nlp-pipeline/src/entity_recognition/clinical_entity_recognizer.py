import re
from typing import Dict, List, Any
from .medical_entity_recognizer import MedicalEntityRecognizer
from model_manager import ModelManager

class ClinicalEntityRecognizer(MedicalEntityRecognizer):
    """Enhanced recognizer for clinical documentation needs"""
    
    def __init__(self, model_manager_instance: ModelManager, model_name: str = "en_core_web_sm"):
        super().__init__(model_manager_instance=model_manager_instance, model_name=model_name)
        self.model_manager = model_manager_instance
        self.entity_types.extend([
            "VITAL_SIGN", "SEVERITY", "FREQUENCY", "DURATION", 
            "CONTRIBUTING_FACTOR", "FAMILY_HISTORY", "NEGATED_SYMPTOM",
            "TREATMENT_EFFICACY", "RECOMMENDATION"
        ])
        
    def extract_clinical_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract structured clinical entities for SOAP notes
        
        Returns:
            Dictionary of categorized entities
        """
        # Get basic entities first
        all_entities = self.extract_entities(text)
        
        # Add specialized clinical entities
        all_entities.extend(self._extract_vital_signs(text))
        all_entities.extend(self._extract_severities(text))
        all_entities.extend(self._extract_contributing_factors(text))
        all_entities.extend(self._extract_negated_findings(text))
        all_entities.extend(self._extract_treatment_efficacy(text))
          # Resolve terminology codes
        from terminology.terminology_resolver import TerminologyResolver
        terminology_resolver = TerminologyResolver(model_manager_instance=self.model_manager)
        # Map entities to standardized terminology
        for entity in all_entities:
            if 'entity_type' in entity and 'text' in entity:
                entity_type = entity['entity_type']
                entity_text = entity['text']
                
                # Resolve entity to standardized terminology
                resolved = terminology_resolver.resolve_entity(entity_text, entity_type)
                if resolved:
                    # Add standardized codes
                    if 'code' in resolved:
                        entity['code'] = resolved['code']
                    if 'code_system' in resolved:
                        entity['code_system'] = resolved['code_system']
                    if 'preferred_term' in resolved:
                        entity['preferred_term'] = resolved['preferred_term']
                    
                    # Add specific code types based on entity type
                    if entity_type == 'MEDICATION' and 'rxnorm_code' in resolved:
                        entity['rxnorm_code'] = resolved['rxnorm_code']
                    elif entity_type in ['CONDITION', 'DIAGNOSIS', 'SYMPTOM']:
                        if 'icd10_code' in resolved:
                            entity['icd10_code'] = resolved['icd10_code']
                        if 'snomed_code' in resolved:
                            entity['snomed_code'] = resolved['snomed_code']
                    elif entity_type == 'LAB_TEST' and 'loinc_code' in resolved:
                        entity['loinc_code'] = resolved['loinc_code']
                    elif entity_type == 'PROCEDURE':
                        if 'cpt_code' in resolved:
                            entity['cpt_code'] = resolved['cpt_code']
                        if 'icd10_code' in resolved:
                            entity['icd10_code'] = resolved['icd10_code']
                        if 'snomed_code' in resolved:
                            entity['snomed_code'] = resolved['snomed_code']
                    
                    # Include all available codes from the resolver
                    for key, value in resolved.items():
                        if key.endswith('_code') and key not in entity:
                            entity[key] = value
        
        # Organize by category
        categorized = {
            "SYMPTOMS": [],
            "NEGATED_SYMPTOMS": [],
            "VITAL_SIGNS": [],
            "SEVERITY": [],
            "CONTRIBUTING_FACTORS": [],
            "FAMILY_HISTORY": [],
            "MEDICATIONS": [],
            "TREATMENT_EFFICACY": [],
            "RECOMMENDATIONS": []
        }
        
        # Categorize entities
        for entity in all_entities:
            entity_type = entity.get('entity_type', '')
            
            if entity_type == 'SYMPTOM':
                categorized["SYMPTOMS"].append(entity)
            elif entity_type == 'NEGATED_SYMPTOM':
                categorized["NEGATED_SYMPTOMS"].append(entity)
            elif entity_type == 'VITAL_SIGN':
                categorized["VITAL_SIGNS"].append(entity)
            elif entity_type == 'SEVERITY':
                categorized["SEVERITY"].append(entity)
            elif entity_type in ['CONTRIBUTING_FACTOR', 'RISK_FACTOR']:
                categorized["CONTRIBUTING_FACTORS"].append(entity)
            elif entity_type == 'FAMILY_HISTORY':
                categorized["FAMILY_HISTORY"].append(entity)
            elif entity_type == 'MEDICATION':
                categorized["MEDICATIONS"].append(entity)
            elif entity_type == 'TREATMENT_EFFICACY':
                categorized["TREATMENT_EFFICACY"].append(entity)
            elif entity_type == 'RECOMMENDATION':
                categorized["RECOMMENDATIONS"].append(entity)
        
        return categorized
        
    def _extract_vital_signs(self, text: str) -> List[Dict[str, Any]]:
        """Extract vital signs like blood pressure, temperature, etc."""
        vital_signs = []
        
        # Blood pressure pattern
        bp_pattern = r'(\d{2,3})[/-](\d{2,3})'
        for match in re.finditer(bp_pattern, text):
            context = text[max(0, match.start()-30):min(len(text), match.end()+30)]
            if 'blood pressure' in context.lower() or 'bp' in context.lower():
                vital_signs.append({
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'entity_type': 'VITAL_SIGN',
                    'vital_type': 'BLOOD_PRESSURE',
                    'systolic': match.group(1),
                    'diastolic': match.group(2),
                    'confidence': 0.9,
                    'method': 'pattern'
                })
        
        return vital_signs
        
    def _extract_severities(self, text: str) -> List[Dict[str, Any]]:
        """Extract severity indicators like pain scales"""
        severities = []
        
        # Pain scale pattern
        pain_scale_pattern = r'(\d{1,2})(?:\s*[-/]?\s*\d{1,2})?\s*(?:out\s*of|\/)\s*10'
        for match in re.finditer(pain_scale_pattern, text, re.IGNORECASE):
            context = text[max(0, match.start()-30):min(len(text), match.end()+30)]
            if 'pain' in context.lower() or 'rate' in context.lower() or 'scale' in context.lower():
                severities.append({
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'entity_type': 'SEVERITY',
                    'severity_type': 'PAIN_SCALE',
                    'value': match.group(1),
                    'max_scale': 10,
                    'confidence': 0.9,
                    'method': 'pattern'
                })
        
        return severities
        
    def _extract_contributing_factors(self, text: str) -> List[Dict[str, Any]]:
        """Extract contributing factors to symptoms/conditions"""
        factors = []
        
        # Key contributing factors for headache/migraine
        factor_patterns = [
            (r'(?:high|too much|lot of)\s*(?:stress|anxiety)', 'STRESS'),
            (r'(?:poor|bad|irregular|not enough)\s*sleep', 'POOR_SLEEP'),
            (r'(?:(\d+)\s*-?\s*(\d+)|several|many|too many|a lot of)\s*(?:cups of coffee|coffee|caffeine)', 'HIGH_CAFFEINE'),
            (r'(?:family|mother|father|parent|sibling).*(?:had|have|history of)\s*(migraines?|headaches?)', 'FAMILY_HISTORY'),
            (r'(?:staying up|up)\s*late\s*(?:at night|nights)', 'POOR_SLEEP_SCHEDULE')
        ]
        
        for pattern, factor_type in factor_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                factors.append({
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'entity_type': 'CONTRIBUTING_FACTOR' if factor_type != 'FAMILY_HISTORY' else 'FAMILY_HISTORY',
                    'factor_type': factor_type,
                    'confidence': 0.8,
                    'method': 'pattern'
                })
                
                # Extract quantity information for caffeine
                if factor_type == 'HIGH_CAFFEINE' and match.groups() and match.group(1):
                    factors[-1]['quantity_low'] = match.group(1)
                    if match.group(2):
                        factors[-1]['quantity_high'] = match.group(2) 
        
        return factors
        
    def _extract_negated_findings(self, text: str) -> List[Dict[str, Any]]:
        """Extract negated symptoms/findings"""
        negated = []
        
        # Common negation patterns
        negation_patterns = [
            r'no\s+(\w+\s*\w*)',
            r'deny\s+(?:any\s+)?(\w+\s*\w*)',
            r'denies\s+(?:any\s+)?(\w+\s*\w*)',
            r'without\s+(?:any\s+)?(\w+\s*\w*)',
            r'negative\s+(?:for\s+)?(\w+\s*\w*)',
            r'not\s+(?:having|experiencing)\s+(?:any\s+)?(\w+\s*\w*)',
            r'never\s+had\s+(?:any\s+)?(\w+\s*\w*)'
        ]
        
        symptom_terms = ['nausea', 'vomiting', 'dizziness', 'fever', 
                        'chills', 'fatigue', 'weakness', 'numbness',
                        'tingling', 'blurry vision', 'headache', 'pain']
        
        for pattern in negation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Check if the negated term is a known symptom
                if match.group(1).lower() in symptom_terms:
                    negated.append({
                        'text': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'entity_type': 'NEGATED_SYMPTOM',
                        'negated_term': match.group(1),
                        'confidence': 0.8,
                        'method': 'pattern'
                    })
        
        return negated
        
    def _extract_treatment_efficacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract information about treatment efficacy"""
        efficacy = []
        
        # Ineffective treatments
        ineffective_patterns = [
            r'(?:tried|taking|took|using|used)\s*(\w+)\s*(?:but|however|though|didn\'t|did not|doesn\'t|does not)\s*(?:help|work|effective|relief)',
            r'(?:\w+)\s*(?:didn\'t|did not|doesn\'t|does not)\s*(?:help|work|effective|provide relief)',
            r'(?:no|not much|not|minimal|little)\s*(?:relief|improvement|effect|help)\s*(?:from|with)\s*(\w+)'
        ]
        
        for pattern in ineffective_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                medication = match.group(1) if match.groups() else "medication"
                efficacy.append({
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'entity_type': 'TREATMENT_EFFICACY',
                    'treatment': medication,
                    'efficacy': 'INEFFECTIVE',
                    'confidence': 0.8,
                    'method': 'pattern'
                })
        
        return efficacy