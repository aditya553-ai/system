from typing import Dict, Any, List

class PromptTemplates:
    """
    A class to hold predefined prompt templates for interacting with LLMs.
    These templates facilitate consistent queries for entity mapping and validation tasks.
    """

    @staticmethod
    def entity_recognition_prompt(entities: str) -> str:
        return (
            "Given the following medical entities, please provide their standardized "
            "terminology and any relevant codes:\n\n"
            f"Entities: {entities}\n\n"
            "Response should include the entity, its standardized name, and any associated codes."
        )

    @staticmethod
    def relation_extraction_prompt(entities: str, context: str) -> str:
        return (
            "Based on the provided entities and context, identify any relationships "
            "between the entities:\n\n"
            f"Entities: {entities}\n"
            f"Context: {context}\n\n"
            "Please list the relationships found, specifying the type of relationship."
        )

    @staticmethod
    def knowledge_graph_validation_prompt(graph_data: Dict[str, Any]) -> str:
        entities = "\n".join([f"- {node['text']} (Type: {node['label']})" 
                             for node in graph_data.get("nodes", [])])
        
        relations = "\n".join([f"- {edge['source_text']} {edge['relation']} {edge['target_text']}" 
                              for edge in graph_data.get("edges", [])])
        
        return (
            "Please validate the medical consistency and accuracy of this knowledge graph.\n\n"
            f"Entities:\n{entities}\n\n"
            f"Relationships:\n{relations}\n\n"
            "Identify any inconsistencies, contradictions, or factually incorrect information. "
            "Return your findings as a JSON object with an 'issues' array containing objects with "
            "'description' and 'severity' (low/medium/high) fields."
        )
    
    @staticmethod
    def entity_mapping_prompt(entity_text: str, entity_type: str, context: str = None) -> str:
        context_section = f"Context: {context}\n\n" if context else ""
        
        return (
            f"Map the medical {entity_type.lower()} '{entity_text}' to standardized medical terminologies.\n\n"
            f"{context_section}"
            "Return your response as a JSON object with the following structure:\n"
            "{\n"
            "  \"preferred_term\": \"standardized term\",\n"
            "  \"code_systems\": {\n"
            "    \"SNOMED\": \"code\",\n"
            "    \"ICD10\": \"code\",\n"
            "    \"RxNORM\": \"code\",\n"
            "    \"LOINC\": \"code\"\n"
            "  },\n"
            "  \"confidence\": 0.95 // value between 0-1\n"
            "}"
        )
    
    @staticmethod
    def medication_mapping_prompt(medication_name: str, context: str = None) -> str:
        context_section = f"Context: {context}\n\n" if context else ""
        
        return (
            f"Map the medication '{medication_name}' to RxNorm standard terminology.\n\n"
            f"{context_section}"
            "Return your response as a JSON object with the following structure:\n"
            "{\n"
            "  \"preferred_term\": \"standardized medication name\",\n"
            "  \"ingredient\": \"active ingredient\", // if applicable\n"
            "  \"code_systems\": {\n"
            "    \"RxNORM\": \"RxCUI code\",\n"
            "    \"ATC\": \"ATC code\" // if available\n"
            "  },\n"
            "  \"confidence\": 0.95 // value between 0-1\n"
            "}"
        )
    
    @staticmethod
    def condition_mapping_prompt(condition_name: str, context: str = None) -> str:
        context_section = f"Context: {context}\n\n" if context else ""
        
        return (
            f"Map the medical condition '{condition_name}' to standardized terminologies.\n\n"
            f"{context_section}"
            "Return your response as a JSON object with the following structure:\n"
            "{\n"
            "  \"preferred_term\": \"standardized condition name\",\n"
            "  \"code_systems\": {\n"
            "    \"SNOMED\": \"SNOMED CT code\",\n"
            "    \"ICD10\": \"ICD-10-CM code\"\n"
            "  },\n"
            "  \"confidence\": 0.95 // value between 0-1\n"
            "}"
        )
    
    @staticmethod
    def lab_test_mapping_prompt(test_name: str, context: str = None) -> str:
        context_section = f"Context: {context}\n\n" if context else ""
        
        return (
            f"Map the laboratory test '{test_name}' to LOINC terminology.\n\n"
            f"{context_section}"
            "Return your response as a JSON object with the following structure:\n"
            "{\n"
            "  \"preferred_term\": \"standardized test name\",\n"
            "  \"code_systems\": {\n"
            "    \"LOINC\": \"LOINC code\"\n"
            "  },\n"
            "  \"specimen\": \"specimen type\", // if applicable\n"
            "  \"confidence\": 0.95 // value between 0-1\n"
            "}"
        )
    
    @staticmethod
    def error_correction_prompt(text: str, speaker_role: str = None) -> str:
        role_context = f" from a {speaker_role}" if speaker_role else ""
        
        return (
            f"Correct any errors in this medical transcript{role_context} while preserving all medical terminology:\n\n"
            f"\"{text}\"\n\n"
            "Focus on fixing misspellings, grammatical errors, and word confusions, but maintain all medical terms "
            "even if they seem unusual. Return only the corrected text without explanations."
        )
    
    @staticmethod
    def graph_refinement_prompt(graph_data: Dict[str, Any]) -> str:
        nodes = "\n".join([f"- {node.get('text', '')} (Type: {node.get('label', '')})" 
                           for node in graph_data.get("nodes", [])])
        
        edges = "\n".join([f"- {edge.get('source_text', '')} {edge.get('relation', 'RELATED_TO')} {edge.get('target_text', '')}" 
                          for edge in graph_data.get("edges", [])])
        
        return (
            "I have a medical knowledge graph with the following entities and relationships:\n\n"
            f"Entities:\n{nodes}\n\n"
            f"Relationships:\n{edges}\n\n"
            "Please identify:\n"
            "1. Missing relationships that should exist between these entities\n"
            "2. Contradictions or incorrect relationships\n"
            "3. Relationships that could be more specific\n\n"
            "Return your suggestions in JSON format with the following structure:\n"
            "{\n"
            "  \"missing_relations\": [\n"
            "    {\"source\": \"entity1\", \"target\": \"entity2\", \"relation\": \"RELATION_TYPE\"}\n"
            "  ],\n"
            "  \"contradictions\": [\n"
            "    {\"source\": \"entity1\", \"target\": \"entity2\", \"explanation\": \"reason for contradiction\"}\n"
            "  ],\n"
            "  \"refinements\": [\n"
            "    {\"source\": \"entity1\", \"target\": \"entity2\", \"old_relation\": \"OLD_TYPE\", \"new_relation\": \"NEW_TYPE\"}\n"
            "  ]\n"
            "}"
        )
    
    @staticmethod
    def medical_contradiction_check_prompt(statement1: str, statement2: str) -> str:
        return (
            "Determine if these two medical statements contradict each other:\n\n"
            f"Statement 1: {statement1}\n"
            f"Statement 2: {statement2}\n\n"
            "Respond with a JSON object containing:\n"
            "{\n"
            "  \"contradiction\": true/false,\n" 
            "  \"explanation\": \"explanation of why they contradict or not\",\n"
            "  \"confidence\": 0.95 // value between 0-1\n"
            "}"
        )
    
    @staticmethod
    def multi_agent_verification_prompt(entity: Dict[str, Any], verification_type: str) -> str:
        entity_text = entity.get("text", "")
        entity_type = entity.get("label", "")
        
        if verification_type == "drug":
            return (
                f"Verify this medication entity: '{entity_text}' (mapped to: {entity.get('preferred_term', entity_text)})\n\n"
                f"RxNorm code: {entity.get('code_systems', {}).get('RXNORM', 'Not provided')}\n\n"
                "Please verify if this is a valid medication and if the mapping is correct. "
                "Return a JSON object with 'verified' (true/false), 'confidence' (0-1), and 'corrected_mapping' if needed."
            )
            
        elif verification_type == "diagnosis":
            return (
                f"Verify this medical condition entity: '{entity_text}' (mapped to: {entity.get('preferred_term', entity_text)})\n\n"
                f"ICD-10 code: {entity.get('code_systems', {}).get('ICD10', 'Not provided')}\n"
                f"SNOMED CT code: {entity.get('code_systems', {}).get('SNOMED', 'Not provided')}\n\n"
                "Please verify if this is a valid medical condition and if the mapping is correct. "
                "Return a JSON object with 'verified' (true/false), 'confidence' (0-1), and 'corrected_mapping' if needed."
            )
            
        elif verification_type == "lab":
            return (
                f"Verify this laboratory test entity: '{entity_text}' (mapped to: {entity.get('preferred_term', entity_text)})\n\n"
                f"LOINC code: {entity.get('code_systems', {}).get('LOINC', 'Not provided')}\n\n"
                "Please verify if this is a valid laboratory test and if the mapping is correct. "
                "Return a JSON object with 'verified' (true/false), 'confidence' (0-1), and 'corrected_mapping' if needed."
            )
            
        else:
            return (
                f"Verify this medical entity of type {entity_type}: '{entity_text}' (mapped to: {entity.get('preferred_term', entity_text)})\n\n"
                "Please verify if this is a valid medical entity and if the mapping is correct. "
                "Return a JSON object with 'verified' (true/false), 'confidence' (0-1), and 'corrected_mapping' if needed."
            )