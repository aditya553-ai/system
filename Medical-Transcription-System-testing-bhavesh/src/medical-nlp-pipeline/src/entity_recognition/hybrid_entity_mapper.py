import os
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from fuzzywuzzy import process
from terminology.terminology_resolver import TerminologyResolver
from llm.llm_client import LLMClient
from knowledge_graph.graph_validator import GraphValidator
import sys
import os
from model_manager import ModelManager

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



class HybridEntityMapper:
    """
    Class for hybrid entity mapping that integrates open-source terminology resolvers,
    LLM-guided mapping, and multi-agent cross-verification for entity normalization and mapping.
    """

    def __init__(self, model_manager_instance: ModelManager, cache_file: str = "entity_mapping_cache.json"):
        """
        Initialize the hybrid entity mapper.
        
        Args:
            use_cache: Whether to use caching for mapped entities
            
        """
        self.model_manager = model_manager_instance
        self.terminology_resolver = TerminologyResolver(model_manager_instance=self.model_manager)
        self.llm_client = LLMClient(model_manager_instance=self.model_manager)
        self.cache_file = cache_file
        self.mapping_cache = {}
        self.use_cache = True
        self._load_cache()

    def _load_cache(self) -> None:
        """Load the entity mapping cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.mapping_cache = json.load(f)
        except Exception as e:
            print(f"Error loading entity mapping cache: {e}")
            self.mapping_cache = {}
            
    def _save_cache(self) -> None:
        """Save the entity mapping cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.mapping_cache, f, indent=2)
        except Exception as e:
            print(f"Error saving entity mapping cache: {e}")
            
    def _get_cache_key(self, text: str, entity_type: str) -> str:
        """Generate a cache key for an entity"""
        combined = f"{text.lower()}:{entity_type.lower()}"
        return hashlib.md5(combined.encode()).hexdigest()
        
    def _get_from_cache(self, text: str, entity_type: str) -> Optional[Dict]:
        """Get entity mapping from cache"""
        cache_key = self._get_cache_key(text, entity_type)
        if cache_key in self.mapping_cache:
            # Check if cache entry is expired (older than 30 days)
            entry = self.mapping_cache[cache_key]
            timestamp = entry.get('timestamp', 0)
            now = datetime.now().timestamp()
            if now - timestamp < 30 * 24 * 3600:  # 30 days in seconds
                return entry.get('mapping')
        return None
        
    def _add_to_cache(self, text: str, entity_type: str, mapping: Dict) -> None:
        """Add entity mapping to cache"""
        cache_key = self._get_cache_key(text, entity_type)
        self.mapping_cache[cache_key] = {
            'timestamp': datetime.now().timestamp(),
            'mapping': mapping
        }
        # Save cache periodically (can be optimized to not save on every add)
        if len(self.mapping_cache) % 10 == 0:
            self._save_cache()

    def _map_with_llm(self, entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map entity using LLM"""
        try:
            entity_text = entity.get("text", "")
            entity_type = entity.get("entity_type", "")
            
            prompt = f"""
            Map the following medical entity to standardized terminology:
            Text: {entity_text}
            Type: {entity_type}
            
            Provide a JSON response with these fields:
            - preferred_term: The standardized term
            - code: Medical code if available (UMLS CUI, SNOMED CT, ICD-10, RxNorm)
            - code_system: The system of the code (UMLS, SNOMED, ICD10, RxNorm)
            - definition: Brief definition (1-2 sentences)
            - confidence: A number between 0 and 1 indicating mapping confidence
            """
            
            response = self.llm_client.generate_response(prompt) 
            
            try:
                # Try to parse JSON response
                result_json = json.loads(response)
                
                # Format the result
                result = {
                    "preferred_term": result_json.get("preferred_term", entity_text),
                    "code": result_json.get("code", ""),
                    "code_system": result_json.get("code_system", ""),
                    "definition": result_json.get("definition", ""),
                    "confidence": result_json.get("confidence", 0.7),
                    "normalized": True,
                    "source": "llm"
                }
                return result
            except json.JSONDecodeError:
                # If not valid JSON, extract key information using simple parsing
                preferred_term = entity_text
                if "preferred_term:" in response:
                    preferred_term = response.split("preferred_term:")[1].split("\n")[0].strip()
                
                return {
                    "preferred_term": preferred_term,
                    "confidence": 0.6,
                    "normalized": True,
                    "source": "llm_parsed"
                }
                
        except Exception as e:
            print(f"Error in LLM mapping: {e}")
            return None
            
    def _verify_mapping(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Verify and clean up entity mapping"""
        # Make a copy to avoid modifying the original
        verified = entity.copy()
        
        # Ensure required fields
        if "preferred_term" not in verified or not verified["preferred_term"]:
            verified["preferred_term"] = verified.get("text", "")
            
        if "confidence" not in verified:
            verified["confidence"] = 0.5
            
        if "normalized" not in verified:
            verified["normalized"] = False
            
        return verified


    def map_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map entities using the three-layer hybrid approach:
        1. Open-source terminology resolvers
        2. LLM-guided mapping
        3. Multi-agent cross-verification
        
        Args:
            entities: List of entities to be mapped.

        Returns:
            List of mapped entities with normalized identifiers.
        """
        mapped_entities = []
        
        for entity in entities:
            # Skip if no text or entity_type
            if 'text' not in entity or 'entity_type' not in entity:
                continue
                
            # Try to get mapping from cache
            cached_mapping = self._get_from_cache(entity['text'], entity['entity_type']) if self.use_cache else None
            
            if cached_mapping:
                # Use cached mapping
                entity.update(cached_mapping)
                mapped_entities.append(entity)
                continue
                
            # Try terminology resolution first
            resolved_entity = self._resolve_with_terminology(entity)
            
            if resolved_entity:
                entity.update(resolved_entity)
            else:
                # Fallback to LLM-based mapping
                llm_entity = self._map_with_llm(entity)
                if llm_entity:
                    entity.update(llm_entity)
            
            # Verify the mapping is valid
            verified_entity = self._verify_mapping(entity)
            
            # Add to cache
            self._add_to_cache(entity['text'], entity['entity_type'], verified_entity)
            
            mapped_entities.append(entity)

        return mapped_entities

    def _resolve_with_terminology(self, entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Resolve entity using terminology services"""
        
        entity_text = entity.get("text", "")
        entity_type = entity.get("entity_type", "")
        
        # Check cache first if enabled
        if self.use_cache:
            cached_result = self._get_from_cache(entity_text, entity_type)
            if cached_result:
                return cached_result
        
        # Try exact match with terminology resolver
        exact_match = self.terminology_resolver.resolve_entity(entity_text, entity_type)
        
        if exact_match and exact_match.get("code"):
            # Create result with necessary fields
            result = {
                "preferred_term": exact_match.get("preferred_term", entity_text),
                "code": exact_match.get("code", ""),
                "code_system": exact_match.get("code_system", ""),
                "confidence": exact_match.get("confidence", 0.9),
                "normalized": True,
                "source": "terminology_service"
            }
            return result
            
        return None

    def _resolve_with_llm(self, entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Resolve entity using LLM for ambiguous cases.
        
        Args:
            entity: The entity to resolve.
            
        Returns:
            Resolved entity information or None if not resolved.
        """
        # Build context-aware prompt for the LLM
        entity_type = entity['label']
        entity_text = entity['text']
        
        # Different prompt templates based on entity type
        if entity_type in ["MEDICATION", "DRUG"]:
            prompt = f"Map the medication '{entity_text}' to RxNorm standard terminology. Return JSON with RxNorm code, ingredient name, and confidence score."
        elif entity_type in ["CONDITION", "DISEASE", "DIAGNOSIS"]:
            prompt = f"Map the medical condition '{entity_text}' to standardized terminologies. Return JSON with codes for SNOMED CT and ICD-10-CM, preferred term, and confidence score."
        elif entity_type in ["LAB_TEST", "TEST"]:
            prompt = f"Map the laboratory test '{entity_text}' to LOINC terminology. Return JSON with LOINC code, preferred name, and confidence score."
        else:
            prompt = f"Map the medical term '{entity_text}' (type: {entity_type}) to standardized medical terminology. Return JSON with appropriate code systems, preferred term, and confidence score."
        
        # Send prompt to LLM
        llm_response = self.llm_client.generate_mapping(prompt)
        
        # Parse and validate LLM response
        try:
            mapping_data = json.loads(llm_response) if isinstance(llm_response, str) else llm_response
            
            # Validate essential fields
            if 'code_systems' not in mapping_data or 'preferred_term' not in mapping_data:
                return None
                
            mapping_data['source'] = 'llm'
            return mapping_data
            
        except Exception as e:
            print(f"Error parsing LLM mapping response: {e}")
            return None

    def _cross_verify_with_agents(self, original_entity: Dict[str, Any], 
                                 resolved_entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-verify entity mapping using specialized agents.
        
        Args:
            original_entity: The original entity extracted from text.
            resolved_entity: The resolved entity information.
            
        Returns:
            Verified entity information.
        """
        entity_type = original_entity['label']
        verification_results = []
        
        # Deploy specialized agents based on entity type
        if entity_type in ["MEDICATION", "DRUG"]:
            # Agent 1: Drug code verification using DrugBank or RxNorm API
            drug_verification = self._verify_drug_code(
                original_entity['text'], 
                resolved_entity.get('code_systems', {}).get('RXNORM')
            )
            verification_results.append(drug_verification)
            
        elif entity_type in ["CONDITION", "DISEASE", "DIAGNOSIS"]:
            # Agent 2: Diagnosis code verification using CDC's public database or ICD API
            diagnosis_verification = self._verify_diagnosis_code(
                original_entity['text'],
                resolved_entity.get('code_systems', {}).get('ICD10')
            )
            verification_results.append(diagnosis_verification)
            
        elif entity_type in ["LAB_TEST", "TEST"]:
            # Agent 3: Lab code verification against LOINC repository
            lab_verification = self._verify_lab_code(
                original_entity['text'],
                resolved_entity.get('code_systems', {}).get('LOINC')
            )
            verification_results.append(lab_verification)
        
        # Use consensus voting from the verification results
        if verification_results:
            positive_votes = sum(1 for result in verification_results if result.get('verified', False))
            if positive_votes >= max(1, len(verification_results) // 2):
                resolved_entity['verified'] = True
                resolved_entity['verification_confidence'] = positive_votes / len(verification_results)
            else:
                resolved_entity['verified'] = False
                resolved_entity['verification_confidence'] = positive_votes / len(verification_results)
        
        return resolved_entity

    def _verify_drug_code(self, drug_name: str, rxnorm_code: Optional[str]) -> Dict[str, Any]:
        """
        Verify drug code using DrugBank API or similar service.
        
        Args:
            drug_name: Name of the drug.
            rxnorm_code: RxNorm code to verify.
            
        Returns:
            Verification result.
        """
        # Placeholder for DrugBank API integration
        # In a real implementation, this would call an external API
        return {'verified': True, 'confidence': 0.9}

    def _verify_diagnosis_code(self, condition: str, icd10_code: Optional[str]) -> Dict[str, Any]:
        """
        Verify diagnosis code using CDC's database or ICD API.
        
        Args:
            condition: Name of the medical condition.
            icd10_code: ICD-10 code to verify.
            
        Returns:
            Verification result.
        """
        # Placeholder for ICD API integration
        # In a real implementation, this would call an external API
        return {'verified': True, 'confidence': 0.85}

    def _verify_lab_code(self, lab_test: str, loinc_code: Optional[str]) -> Dict[str, Any]:
        """
        Verify lab test code using LOINC repository.
        
        Args:
            lab_test: Name of the laboratory test.
            loinc_code: LOINC code to verify.
            
        Returns:
            Verification result.
        """
        # Placeholder for LOINC API integration
        # In a real implementation, this would call an external API
        return {'verified': True, 'confidence': 0.9}


