import os
import json
import requests
import logging
import time
from typing import List, Dict, Any, Optional, Union
from .prompt_templates import PromptTemplates
from dotenv import load_dotenv
from model_manager import ModelManager

load_dotenv()

class LLMClient:
    """
    LLMClient class for interfacing with various LLMs for entity mapping and validation tasks.
    """

    def __init__(self, 
                 model_manager_instance: ModelManager,
                 api_key: str = None, 
                 model: str = "meta/llama3-70b-instruct",  # Updated model
                 base_url: str = "https://integrate.api.nvidia.com/v1",  # Updated URL
                 max_retries: int = 3,
                 retry_delay: int = 2):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key for accessing LLM services (default: from environment)
            model: Which model to use (default: meta/llama3-70b-instruct via NVIDIA)
            base_url: API endpoint URL (default: NVIDIA API)
            max_retries: Maximum number of retry attempts on error
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.api_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
        # Set headers for NVIDIA API
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Initialize prompt templates
        self.prompt_templates = PromptTemplates()
        self.templates = self.prompt_templates

    def generate_relation_enhancement(self, entities, existing_relations=[]):
        """
        Use the LLM to enhance relations between entities through text generation.
        
        Args:
            entities: List of medical entities
            existing_relations: Any relations already identified
            
        Returns:
            List of enhanced or additional relations
        """
        try:
            # Convert entities to a string representation for the prompt
            entity_text = "\n".join([f"- {e.get('text', '')} ({e.get('entity_type', 'UNKNOWN')})" 
                                for e in entities if isinstance(e, dict)])
            
            # Describe existing relations
            relations_text = "None"
            if existing_relations:
                relations_text = "\n".join([
                    f"- {r.get('source', '')} {r.get('relation_type', 'RELATED_TO')} {r.get('target', '')}"
                    for r in existing_relations if isinstance(r, dict)
                ])
            
            # Create the prompt
            prompt = f"""
            Analyze these medical entities and identify meaningful clinical relationships between them:
            
            ENTITIES:
            {entity_text}
            
            EXISTING IDENTIFIED RELATIONSHIPS:
            {relations_text}
            
            Identify additional clinically relevant relationships between these entities. 
            Focus on treatment relationships, causal relationships, symptomatic relationships, 
            diagnostic relationships, and anatomical relationships.
            
            For each relationship, provide:
            1. Source entity
            2. Target entity 
            3. Relationship type (TREATS, CAUSES, INDICATES, DIAGNOSES, LOCATED_IN, etc.)
            4. Confidence level (0-1)
            
            Return your answer as a JSON array of relationships in this format:
            [
                {{"source": "entity1", "target": "entity2", "relation_type": "RELATIONSHIP", "confidence": 0.9}}
            ]
            """
            
            # Generate response using existing generate_response method
            response = self.generate_response(prompt)
            if not response:
                return self._generate_basic_relations(entities)
                
            # Extract JSON from the response
            import re
            import json
            
            # Look for JSON array in the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if json_match:
                try:
                    relations_json = json.loads(json_match.group(0))
                    if isinstance(relations_json, list):
                        return relations_json
                except json.JSONDecodeError:
                    pass
                    
            # If we couldn't find a proper JSON array, try to extract individual objects
            json_objects = re.findall(r'\{\s*"source"\s*:.*?\}', response, re.DOTALL)
            relations = []
            for obj_str in json_objects:
                try:
                    rel_obj = json.loads(obj_str)
                    if isinstance(rel_obj, dict) and "source" in rel_obj and "target" in rel_obj:
                        relations.append(rel_obj)
                except json.JSONDecodeError:
                    continue
                    
            if relations:
                return relations
                
            # If all parsing fails, fall back to basic relations
            return self._generate_basic_relations(entities)
                
        except Exception as e:
            print(f"Error enhancing relations with LLM: {e}")
            return self._generate_basic_relations(entities)
                
    def _generate_basic_relations(self, entities):
        """
        Generate basic relations between entities as a fallback
        when API calls fail.
        """
        relations = []
        
        # Map entities by type
        entities_by_type = {}
        for entity in entities:
            if isinstance(entity, dict):
                entity_type = entity.get("entity_type", "")
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(entity)
        
        # Generate basic relations based on entity types
        if "MEDICATION" in entities_by_type and "CONDITION" in entities_by_type:
            for med in entities_by_type["MEDICATION"]:
                for cond in entities_by_type["CONDITION"]:
                    relations.append({
                        "source": med.get("text", ""),
                        "target": cond.get("text", ""),
                        "relation_type": "TREATS",
                        "confidence": 0.5
                    })
                    
        if "SYMPTOM" in entities_by_type and "CONDITION" in entities_by_type:
            for symp in entities_by_type["SYMPTOM"]:
                for cond in entities_by_type["CONDITION"]:
                    relations.append({
                        "source": symp.get("text", ""),
                        "target": cond.get("text", ""),
                        "relation_type": "INDICATES",
                        "confidence": 0.6
                    })
        
        return relations

    def generate_response(self, 
                         prompt: str, 
                         system_prompt: str = None,
                         temperature: float = 0.1, 
                         max_tokens: int = 1024) -> Optional[str]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt to send to the LLM
            system_prompt: Optional system prompt to guide the model's behavior
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in the response
            
        Returns:
            Generated text response or None if failed
        """
        if not self.api_key:
            self.logger.warning("No API key provided. LLM functionality is disabled.")
            return None

        # Create messages for NVIDIA API format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Set up request payload for NVIDIA API
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Make API request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract the generated text from the response
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    self.logger.error(f"Unexpected response structure: {result}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"API request attempt {attempt+1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"Failed after {self.max_retries} attempts")
                    return None

    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from a response string"""
        try:
            # Find JSON-like content between curly braces
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            return None
        except Exception:
            return None

    def map_entities(self, entity_text: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """
        Map a medical entity to standardized terminology using LLM.
        
        Args:
            entity_text: The text of the entity to map
            entity_type: The type of entity (medication, condition, etc.)
            
        Returns:
            Dictionary with mapping information or None if failed
        """
        # Get the appropriate prompt for entity mapping
        prompt = self.prompt_templates.get_entity_mapping_prompt(entity_text, entity_type)
        
        # Generate response
        system_prompt = "You are a medical terminology expert that maps medical terms to standard codes."
        response = self.generate_response(prompt, system_prompt=system_prompt, temperature=0.1)
        
        if not response:
            return None
            
        try:
            # Try to extract JSON from the response
            result = self._extract_json_from_response(response)
            if not result:
                # Fallback to structured response parsing
                return self._parse_structured_response(response, entity_text, entity_type)
                
            return result
        except Exception as e:
            self.logger.error(f"Failed to parse entity mapping response: {str(e)}")
            return None

    def validate_mapping(self, entity_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate the entity mapping using the LLM.

        Args:
            entity_mapping: A dictionary of entities and their mapped identifiers.

        Returns:
            A dictionary containing validation results.
        """
        # Format the mapping as a string for the prompt
        mapping_str = "\n".join([f"- {entity}: {mapping}" for entity, mapping in entity_mapping.items()])
        
        prompt = f"Validate the correctness of these medical entity mappings:\n\n{mapping_str}\n\n"
        prompt += "For each mapping, indicate if it's correct, and if not, provide the correct mapping."
        
        return self._send_request(
            endpoint="/validate_mapping" if "/v1" not in self.api_url else "/completions",
            payload={
                "model": self.model,
                "prompt": prompt,
                "temperature": 0.2,
                "max_tokens": 500
            }
        )

    def cross_verify_entities(self, entities: List[Dict[str, Any]], 
                           verification_types: List[str]) -> List[Dict[str, Any]]:
        """
        Cross-verify entities with multiple agents.

        Args:
            entities: A list of entities to verify.
            verification_types: A list of verification types to apply.

        Returns:
            A list containing the verification results for each entity.
        """
        results = []
        
        for entity in entities:
            entity_results = []
            
            # Apply each verification type
            for vtype in verification_types:
                if vtype in ["drug", "diagnosis", "lab"]:
                    prompt = self.templates.multi_agent_verification_prompt(entity, vtype)
                    
                    response = self._send_request(
                        endpoint="/cross_verify" if "/v1" not in self.api_url else "/completions",
                        payload={
                            "model": self.model,
                            "prompt": prompt,
                            "temperature": 0.2,
                            "max_tokens": 300
                        }
                    )
                    
                    entity_results.append({
                        "verification_type": vtype,
                        "result": response
                    })
            
            # Combine verification results for this entity
            consensus = self._determine_consensus(entity_results)
            
            results.append({
                "entity": entity,
                "verifications": entity_results,
                "consensus": consensus
            })
            
        return results

    def _determine_consensus(self, verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine consensus from multiple verification results.
        
        Args:
            verification_results: List of verification results
            
        Returns:
            Consensus decision
        """
        verified_count = 0
        total_confidence = 0
        corrections = []
        
        for vresult in verification_results:
            result_data = vresult.get("result", {})
            
            if result_data.get("verified", False):
                verified_count += 1
                total_confidence += result_data.get("confidence", 0.5)
            
            if "corrected_mapping" in result_data:
                corrections.append(result_data["corrected_mapping"])
        
        # Determine consensus verification
        if verified_count > len(verification_results) / 2:
            return {
                "verified": True,
                "confidence": total_confidence / verified_count if verified_count > 0 else 0.0,
                "corrections": corrections if corrections else None
            }
        else:
            return {
                "verified": False,
                "confidence": 0.0,
                "corrections": corrections if corrections else None
            }

    def generate_mapping(self, prompt: str) -> Dict[str, Any]:
        """
        Generate entity mapping based on a custom prompt.
        
        Args:
            prompt: Custom prompt for entity mapping
            
        Returns:
            Mapping data
        """
        try:
            response = self._send_request(
                endpoint="/completions" if "openai" in self.api_url else "/generate",
                payload={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": 0.2,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"}
                }
            )
            
            # Extract JSON from response
            if isinstance(response, str):
                # Try to extract JSON from the response string
                try:
                    start_idx = response.find('{')
                    end_idx = response.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = response[start_idx:end_idx+1]
                        return json.loads(json_str)
                    else:
                        return {"error": "JSON not found in response"}
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON in response", "raw_response": response}
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating mapping: {str(e)}")
            return {"error": str(e)}

    def validate_medical_knowledge(self, prompt: str) -> Dict[str, Any]:
        """
        Validate medical knowledge using the LLM.
        
        Args:
            prompt: Validation prompt
            
        Returns:
            Validation results
        """
        try:
            response = self._send_request(
                endpoint="/completions" if "openai" in self.api_url else "/validate",
                payload={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": 0.1,
                    "max_tokens": 800,
                    "response_format": {"type": "json_object"}
                }
            )
            
            # Handle string responses (extract JSON)
            if isinstance(response, str):
                try:
                    start_idx = response.find('{')
                    end_idx = response.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = response[start_idx:end_idx+1]
                        return json.loads(json_str)
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON in response", "issues": []}
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error validating medical knowledge: {str(e)}")
            return {"error": str(e), "issues": []}

    def generate_graph_refinement(self, prompt: str) -> Dict[str, Any]:
        """
        Generate graph refinement suggestions using the LLM.
        
        Args:
            prompt: Graph refinement prompt
            
        Returns:
            Refinement suggestions
        """
        try:
            response = self._send_request(
                endpoint="/completions" if "openai" in self.api_url else "/refine",
                payload={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": 0.2,
                    "max_tokens": 800,
                    "response_format": {"type": "json_object"}
                }
            )
            
            # Handle string responses
            if isinstance(response, str):
                try:
                    start_idx = response.find('{')
                    end_idx = response.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = response[start_idx:end_idx+1]
                        return json.loads(json_str)
                except json.JSONDecodeError:
                    return {"missing_relations": [], "contradictions": [], "refinements": []}
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating graph refinement: {str(e)}")
            return {"error": str(e), "missing_relations": [], "contradictions": [], "refinements": []}

    def suggest_graph_improvements(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Suggest improvements for a knowledge graph.
        
        Args:
            prompt: Graph improvement prompt
            
        Returns:
            List of improvement suggestions
        """
        try:
            response = self._send_request(
                endpoint="/completions" if "openai" in self.api_url else "/improve",
                payload={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": 0.3,
                    "max_tokens": 1000,
                    "response_format": {"type": "json_object"}
                }
            )
            
            # Extract suggestions from response
            if isinstance(response, dict) and "suggestions" in response:
                return response["suggestions"]
            elif isinstance(response, list):
                return response
            
            # Handle string responses
            if isinstance(response, str):
                try:
                    # Try to extract JSON array or object
                    if response.strip().startswith("["):
                        end_idx = response.rfind(']')
                        if end_idx != -1:
                            json_str = response[:end_idx+1]
                            return json.loads(json_str)
                    elif response.strip().startswith("{"):
                        start_idx = response.find('{')
                        end_idx = response.rfind('}')
                        if start_idx != -1 and end_idx != -1:
                            json_str = response[start_idx:end_idx+1]
                            obj = json.loads(json_str)
                            return obj.get("suggestions", [])
                except json.JSONDecodeError:
                    return []
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error suggesting graph improvements: {str(e)}")
            return []

    def correct_text(self, text: str, speaker_role: Optional[str] = None) -> str:
        """
        Correct text using the LLM.
        
        Args:
            text: Text to correct
            speaker_role: Optional role of the speaker
            
        Returns:
            Corrected text
        """
        prompt = self.templates.error_correction_prompt(text, speaker_role)
        
        try:
            response = self._send_request(
                endpoint="/completions" if "openai" in self.api_url else "/correct",
                payload={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": 0.2,
                    "max_tokens": min(len(text) * 2, 1000)
                }
            )
            
            if isinstance(response, dict) and "text" in response:
                return response["text"]
            elif isinstance(response, str):
                return response
                
            return text  # Return original if something went wrong
            
        except Exception as e:
            self.logger.error(f"Error correcting text: {str(e)}")
            return text

    def _send_request(self, endpoint: str, payload: Dict[str, Any]) -> Union[Dict[str, Any], str]:
        """
        Send a request to the LLM API.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            
        Returns:
            API response
        """
        try:
            # Handle OpenAI API format
            if "openai" in self.api_url and "/v1" in self.api_url:
                if endpoint == "/completions":
                    # For text-based models like GPT-4
                    if payload.get("response_format", {}).get("type") == "json_object":
                        endpoint = "/chat/completions"
                        messages = [{"role": "user", "content": payload.pop("prompt")}]
                        payload["messages"] = messages
                    else:
                        endpoint = "/chat/completions"
                        messages = [{"role": "user", "content": payload.pop("prompt")}]
                        payload["messages"] = messages
            
            # Send the request
            url = f"{self.api_url}{endpoint}"
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            # Check for errors
            if response.status_code != 200:
                self.logger.error(f"API error: {response.status_code}, {response.text}")
                return {"error": f"API error: {response.status_code}", "details": response.text}
            
            # Parse the response
            data = response.json()
            
            # Handle different API response formats
            if "openai" in self.api_url:
                if "choices" in data:
                    if "message" in data["choices"][0]:  # Chat completion
                        return data["choices"][0]["message"]["content"]
                    else:  # Regular completion
                        return data["choices"][0]["text"]
            
            # Default response handling
            return data.get("result", data)
            
        except requests.RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {str(e)}")
            return {"error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}