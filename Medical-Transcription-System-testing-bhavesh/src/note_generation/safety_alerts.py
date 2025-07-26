import os
import json
import requests
from typing import Dict, List, Any, Optional
import re
import time

class SafetyAlertGenerator:
    """
    Class for generating safety alerts from clinical note data
    by checking medication interactions, allergies, and contraindications
    """
    
    def __init__(self, config_path=None):
        """Initialize with configuration file path"""
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Load knowledge base for common interactions
        self.interaction_db = self._load_interaction_db()
        self.condition_contraindications = self._load_contraindications()
        
        # Cache for API calls
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "safety_alerts_cache.json")
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load API cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save API cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Error saving safety alerts cache: {e}")
    
    def _load_interaction_db(self):
        """Load common drug-drug interactions database"""
        # This would typically load from a comprehensive database
        # For this implementation, we'll use a small set of common interactions
        return {
            "metformin": {
                "lisinopril": {
                    "severity": "low",
                    "description": "Monitor for hypoglycemia. ACE inhibitors may enhance the hypoglycemic effect of metformin."
                },
                "ibuprofen": {
                    "severity": "moderate",
                    "description": "NSAIDs may increase risk of lactic acidosis with metformin. Monitor renal function."
                },
                "alcohol": {
                    "severity": "high",
                    "description": "Alcohol increases risk of lactic acidosis with metformin. Avoid alcohol consumption."
                }
            },
            "insulin": {
                "beta-blockers": {
                    "severity": "moderate",
                    "description": "Beta-blockers may mask symptoms of hypoglycemia and prolong hypoglycemic effects of insulin."
                },
                "ace inhibitors": {
                    "severity": "low",
                    "description": "ACE inhibitors may enhance the hypoglycemic effect of insulin."
                }
            },
            "lisinopril": {
                "potassium supplements": {
                    "severity": "moderate",
                    "description": "Increased risk of hyperkalemia when ACE inhibitors are used with potassium supplements."
                },
                "nsaids": {
                    "severity": "moderate",
                    "description": "NSAIDs may diminish the antihypertensive effect of ACE inhibitors."
                }
            }
        }
    
    def _load_contraindications(self):
        """Load common medication-condition contraindications"""
        # This would typically load from a comprehensive database
        # For this implementation, we'll use a small set of common contraindications
        return {
            "metformin": {
                "renal failure": {
                    "severity": "high",
                    "description": "Metformin is contraindicated in patients with significant renal impairment due to increased risk of lactic acidosis."
                },
                "liver disease": {
                    "severity": "high",
                    "description": "Metformin should be avoided in patients with liver disease due to increased risk of lactic acidosis."
                }
            },
            "insulin": {
                "hypoglycemia": {
                    "severity": "high",
                    "description": "Use caution with insulin in patients with frequent hypoglycemic episodes."
                }
            },
            "lisinopril": {
                "pregnancy": {
                    "severity": "high",
                    "description": "ACE inhibitors are contraindicated in pregnancy due to risk of fetal injury and death."
                },
                "angioedema": {
                    "severity": "high",
                    "description": "ACE inhibitors are contraindicated in patients with a history of angioedema."
                }
            }
        }
    
    def generate_alerts(self, clinical_note: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate safety alerts based on the clinical note
        
        Args:
            clinical_note: Structured clinical note with medications, conditions, etc.
            
        Returns:
            List of safety alerts with severity, description, etc.
        """
        alerts = []
        
        # Extract medications from the plan section
        medications = self._extract_medications(clinical_note)
        conditions = self._extract_conditions(clinical_note)
        allergies = self._extract_allergies(clinical_note)
        
        # Check for drug-drug interactions
        drug_interactions = self._check_drug_interactions(medications)
        alerts.extend(drug_interactions)
        
        # Check for contraindications based on conditions
        contraindications = self._check_contraindications(medications, conditions)
        alerts.extend(contraindications)
        
        # Check for allergies
        allergy_alerts = self._check_allergies(medications, allergies)
        alerts.extend(allergy_alerts)
        
        # Add timestamps to alerts
        for alert in alerts:
            alert["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        return alerts
    
    def _extract_medications(self, clinical_note: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract medications from the clinical note"""
        medications = []
        
        # Check in the plan section
        if "plan" in clinical_note:
            if "medications" in clinical_note["plan"]:
                medications.extend(clinical_note["plan"]["medications"])
        
        # Check if there's a dedicated medications section
        if "medications" in clinical_note:
            medications.extend(clinical_note["medications"])
        
        return medications
    
    def _extract_conditions(self, clinical_note: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract conditions from the clinical note"""
        conditions = []
        
        # Check in the assessment section
        if "assessment" in clinical_note:
            if "diagnoses" in clinical_note["assessment"]:
                conditions.extend(clinical_note["assessment"]["diagnoses"])
        
        # Check if there's a dedicated conditions section
        if "conditions" in clinical_note:
            conditions.extend(clinical_note["conditions"])
        
        # Check patient history
        if "subjective" in clinical_note:
            if "history" in clinical_note["subjective"]:
                history = clinical_note["subjective"]["history"]
                if "medical_history" in history:
                    conditions.extend(history["medical_history"])
        
        return conditions
    
    def _extract_allergies(self, clinical_note: Dict[str, Any]) -> List[str]:
        """Extract allergies from the clinical note"""
        allergies = []
        
        # Check in the subjective section
        if "subjective" in clinical_note:
            if "allergies" in clinical_note["subjective"]:
                allergies.extend(clinical_note["subjective"]["allergies"])
        
        # Check if there's a dedicated allergies section
        if "allergies" in clinical_note:
            allergies.extend(clinical_note["allergies"])
        
        # Check patient info
        if "patient_info" in clinical_note:
            if "allergies" in clinical_note["patient_info"]:
                allergies.extend(clinical_note["patient_info"]["allergies"])
        
        return allergies
    
    def _check_drug_interactions(self, medications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for drug-drug interactions"""
        alerts = []
        
        # Get medication names
        med_names = [med.get("name", "").lower() for med in medications if "name" in med]
        
        # Check each pair of medications for interactions
        for i, med1 in enumerate(med_names):
            med1_lower = self._normalize_medication_name(med1)
            
            for med2 in med_names[i+1:]:
                med2_lower = self._normalize_medication_name(med2)
                
                # Check local database
                interaction = self._check_local_interaction_db(med1_lower, med2_lower)
                
                if interaction:
                    alerts.append({
                        "type": "drug_interaction",
                        "severity": interaction.get("severity", "moderate"),
                        "description": interaction.get("description", "Potential drug interaction detected."),
                        "medications": [med1, med2]
                    })
                else:
                    # If not in local database, could check online API
                    # But we'll skip for this implementation
                    pass
        
        return alerts
    
    def _normalize_medication_name(self, name: str) -> str:
        """Normalize medication name for comparison"""
        return re.sub(r'\s+', ' ', name.lower().strip())
    
    def _check_local_interaction_db(self, med1: str, med2: str) -> Optional[Dict[str, Any]]:
        """Check local database for drug interactions"""
        # Check if med1 is in the database and has an interaction with med2
        if med1 in self.interaction_db and med2 in self.interaction_db[med1]:
            return self.interaction_db[med1][med2]
        
        # Check if med2 is in the database and has an interaction with med1
        if med2 in self.interaction_db and med1 in self.interaction_db[med2]:
            return self.interaction_db[med2][med1]
        
        return None
    
    def _check_contraindications(self, medications: List[Dict[str, Any]], conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for medication contraindications based on patient conditions"""
        alerts = []
        
        # Get medication names
        med_names = [med.get("name", "").lower() for med in medications if "name" in med]
        
        # Get condition names
        condition_names = [cond.get("name", "").lower() for cond in conditions if "name" in cond]
        
        # Check each medication against each condition
        for med in med_names:
            med_lower = self._normalize_medication_name(med)
            
            for condition in condition_names:
                condition_lower = condition.lower().strip()
                
                # Check if this medication is contraindicated for this condition
                if med_lower in self.condition_contraindications:
                    if condition_lower in self.condition_contraindications[med_lower]:
                        contraindication = self.condition_contraindications[med_lower][condition_lower]
                        
                        alerts.append({
                            "type": "contraindication",
                            "severity": contraindication.get("severity", "high"),
                            "description": contraindication.get("description", 
                                          f"Potential contraindication between {med} and condition {condition}."),
                            "medication": med,
                            "condition": condition
                        })
        
        return alerts
    
    def _check_allergies(self, medications: List[Dict[str, Any]], allergies: List[str]) -> List[Dict[str, Any]]:
        """Check for medication allergies"""
        alerts = []
        
        # If no allergies, return empty list
        if not allergies:
            return []
        
        # Get medication names
        med_names = [med.get("name", "").lower() for med in medications if "name" in med]
        
        # Normalize allergies
        normalized_allergies = [allergy.lower().strip() for allergy in allergies if allergy]
        
        # Check each medication against allergies
        for med in med_names:
            med_lower = self._normalize_medication_name(med)
            
            for allergy in normalized_allergies:
                # Check for exact matches or if medication contains the allergy name
                if med_lower == allergy or med_lower in allergy or allergy in med_lower:
                    alerts.append({
                        "type": "allergy",
                        "severity": "high",
                        "description": f"Patient has a documented allergy to {allergy}, which may conflict with {med}.",
                        "medication": med,
                        "allergy": allergy
                    })
        
        return alerts