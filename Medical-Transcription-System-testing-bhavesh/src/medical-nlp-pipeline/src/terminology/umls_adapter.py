from typing import List, Dict, Any, Optional
import requests
import os
import json
import time
import logging
from fuzzywuzzy import process

class UMLSAdapter:
    """
    Adapter for interacting with the UMLS Metathesaurus.
    Facilitates access to standardized medical codes and terminology.
    """

    def __init__(self, api_key: str = None, umls_api_url: str = "https://uts-ws.nlm.nih.gov/rest"):
        """
        Initialize the UMLS adapter.

        Args:
            api_key: API key for accessing UMLS services.
            umls_api_url: Base URL for UMLS API.
        """
        self.api_key = api_key or os.environ.get("UMLS_API_KEY", "")
        self.umls_api_url = umls_api_url
        self.logger = logging.getLogger(__name__)
        self.token = None
        
        # Initialize only if API key is provided
        if self.api_key:
            self.token = self.get_token()
            
        # Fallback resolver for when UMLS is not available
        self.fallback_resolver = None
        
        # Cache for UMLS lookups
        self.cache = {}

    def get_token(self) -> Optional[str]:
        """
        Retrieve the UMLS API token.

        Returns:
            str: The access token for UMLS API.
        """
        if not self.api_key:
            self.logger.warning("No UMLS API key provided. UMLS functionality will be limited.")
            return None
            
        try:
            url = f"{self.umls_api_url}/auth/token"
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }
            data = {
                "apikey": self.api_key,
                "grant_type": "client_credentials"
            }
            response = requests.post(url, headers=headers, data=data, timeout=10)
            response.raise_for_status()
            return response.json().get("access_token")
        except Exception as e:
            self.logger.error(f"Failed to get UMLS token: {e}")
            return None

    def search_term(self, term: str) -> List[Dict[str, Any]]:
        """
        Search for a medical term in the UMLS Metathesaurus.

        Args:
            term: The medical term to search for.

        Returns:
            List[Dict[str, Any]]: A list of matching concepts.
        """
        # Check cache first
        cache_key = f"search:{term.lower()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # If no token, use fallback resolver
        if not self.token:
            return self._fallback_search(term)
            
        try:
            url = f"{self.umls_api_url}/search/current"
            params = {
                "string": term,
                "ticket": self.token,
                "pageSize": 10
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            results = response.json().get("result", {}).get("results", [])
            
            # Cache results
            self.cache[cache_key] = results
            return results
        except Exception as e:
            self.logger.error(f"UMLS search error for '{term}': {e}")
            return self._fallback_search(term)

    def get_concept_details(self, cui: str) -> Dict[str, Any]:
        """
        Retrieve detailed information about a concept using its CUI.

        Args:
            cui: The Concept Unique Identifier (CUI) of the concept.

        Returns:
            Dict[str, Any]: Detailed information about the concept.
        """
        # Check cache first
        cache_key = f"concept:{cui}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # If no token, return minimal info
        if not self.token:
            return {"cui": cui, "error": "No UMLS token available"}
            
        try:
            url = f"{self.umls_api_url}/content/current/CUI/{cui}"
            params = {
                "ticket": self.token
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            
            # Cache results
            self.cache[cache_key] = result
            return result
        except Exception as e:
            self.logger.error(f"UMLS concept detail error for '{cui}': {e}")
            return {"cui": cui, "error": str(e)}

    def validate_concept(self, concept: Dict[str, Any]) -> bool:
        """
        Validate a concept against UMLS standards.

        Args:
            concept: The concept dictionary to validate.

        Returns:
            bool: True if the concept is valid, False otherwise.
        """
        # Basic validation - check for required fields
        if not concept:
            return False
            
        # Check for CUI or equivalent identifier
        has_identifier = any(key in concept for key in ["cui", "CUI", "conceptId", "id", "code"])
        
        # Check for name or equivalent field
        has_name = any(key in concept for key in ["name", "label", "prefLabel", "preferredName"])
        
        return has_identifier and has_name

    def map_to_standard_codes(self, term: str) -> List[Dict[str, Any]]:
        """
        Map a medical term to standardized codes using UMLS.

        Args:
            term: The medical term to map.

        Returns:
            List[Dict[str, Any]]: A list of standardized codes and their details.
        """
        concepts = self.search_term(term)
        mapped_codes = []
        
        for concept in concepts:
            if self.validate_concept(concept):
                cui = concept.get("ui", concept.get("conceptId", concept.get("CUI")))
                
                if cui:
                    details = self.get_concept_details(cui)
                    
                    # Extract relevant code mappings
                    code_mappings = self._extract_code_mappings(details)
                    
                    mapped_codes.append({
                        "cui": cui,
                        "name": concept.get("name", concept.get("label", term)),
                        "source": "UMLS",
                        "codes": code_mappings
                    })
        
        # If no UMLS results, use fallback resolver
        if not mapped_codes and self.fallback_resolver:
            return self._fallback_map_to_codes(term)
            
        return mapped_codes

    def _extract_code_mappings(self, concept_details: Dict[str, Any]) -> Dict[str, str]:
        """Extract standardized code mappings from concept details"""
        code_mappings = {}
        
        # Extract codes from UMLS API response
        atoms = concept_details.get("atoms", [])
        for atom in atoms:
            source = atom.get("rootSource")
            code = atom.get("code")
            
            if source and code:
                # Map to standard code systems
                if source == "SNOMEDCT_US":
                    code_mappings["SNOMED"] = code
                elif source == "ICD10CM":
                    code_mappings["ICD10"] = code
                elif source == "RXNORM":
                    code_mappings["RxNORM"] = code
                elif source == "LNC":
                    code_mappings["LOINC"] = code
                else:
                    code_mappings[source] = code
        
        return code_mappings

    def _fallback_search(self, term: str) -> List[Dict[str, Any]]:
        """Use fallback resolver when UMLS is not available"""
        # Determine entity type based on term
        entity_type = self._guess_entity_type(term)
        
        # Resolve using fallback resolver
        result = self.fallback_resolver.resolve_entity(term, entity_type)
        
        if result:
            return [{
                "ui": result.get("code", ""),
                "name": result.get("preferred_term", term),
                "rootSource": result.get("code_system", ""),
                "uri": "",
                "fallback": True
            }]
        return []

    def _fallback_map_to_codes(self, term: str) -> List[Dict[str, Any]]:
        """Use fallback resolver for code mapping"""
        entity_type = self._guess_entity_type(term)
        result = self.fallback_resolver.resolve_entity(term, entity_type)
        
        if result:
            # Convert to format similar to UMLS response
            codes = {}
            
            # Extract codes based on entity type
            if entity_type in ["MEDICATION", "DRUG"]:
                if "rxnorm_code" in result:
                    codes["RxNORM"] = result["rxnorm_code"]
            elif entity_type in ["CONDITION", "DIAGNOSIS"]:
                if "icd10_code" in result:
                    codes["ICD10"] = result["icd10_code"]
                if "snomed_code" in result:
                    codes["SNOMED"] = result["snomed_code"]
            elif entity_type in ["LAB_TEST", "TEST"]:
                if "loinc_code" in result:
                    codes["LOINC"] = result["loinc_code"]
            
            return [{
                "cui": result.get("code", ""),
                "name": result.get("preferred_term", term),
                "source": result.get("source", "fallback"),
                "codes": codes,
                "fallback": True
            }]
        return []

    def _guess_entity_type(self, term: str) -> str:
        """Guess the entity type based on the term"""
        term_lower = term.lower()
        
        # Simple heuristics for guessing entity type
        medication_indicators = ["mg", "tablet", "capsule", "injection", "pill", "dose", "spray"]
        lab_indicators = ["test", "level", "count", "measurement", "panel", "screening"]
        
        for indicator in medication_indicators:
            if indicator in term_lower:
                return "MEDICATION"
                
        for indicator in lab_indicators:
            if indicator in term_lower:
                return "LAB_TEST"
        
        # Default to condition if no other indicators found
        return "CONDITION"

    def get_semantic_type(self, cui: str) -> str:
        """
        Get the semantic type of a concept.

        Args:
            cui: The CUI of the concept.

        Returns:
            str: The semantic type of the concept.
        """
        if not self.token:
            return "unknown"
            
        try:
            url = f"{self.umls_api_url}/content/current/CUI/{cui}/semanticTypes"
            params = {
                "ticket": self.token
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            semantic_types = response.json().get("result", [])
            
            if semantic_types:
                return semantic_types[0].get("name", "unknown")
            return "unknown"
        except Exception as e:
            self.logger.error(f"Error getting semantic type for {cui}: {e}")
            return "unknown"

    def get_relationships(self, cui: str) -> List[Dict[str, Any]]:
        """
        Get relationships for a concept.

        Args:
            cui: The CUI of the concept.

        Returns:
            List[Dict[str, Any]]: The relationships of the concept.
        """
        if not self.token:
            return []
            
        try:
            url = f"{self.umls_api_url}/content/current/CUI/{cui}/relations"
            params = {
                "ticket": self.token
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json().get("result", [])
        except Exception as e:
            self.logger.error(f"Error getting relationships for {cui}: {e}")
            return []

    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()