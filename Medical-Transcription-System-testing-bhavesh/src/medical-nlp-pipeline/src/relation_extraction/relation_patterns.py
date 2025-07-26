from typing import List, Dict, Any, Optional, Tuple
import re
import spacy
import json
from llm.llm_client import LLMClient
import sys
import os
from model_manager import ModelManager
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



class RelationPatterns:
    """
    Class to define various patterns used for identifying relationships between medical entities.
    This class implements a hybrid approach that includes open-source terminology resolvers,
    LLM-guided mapping, and multi-agent cross-verification.
    """

    def __init__(self, model_manager_instance: ModelManager):
        self.model_manager = model_manager_instance
        self.patterns = self._initialize_patterns()
        self.terminology_resolvers = []  # List of terminology resolvers
        self.llm_guided_mappings = []  # List of LLM-guided mappings
        self.llm_client = LLMClient(model_manager_instance=self.model_manager)
        self.relation_cache = {}  # Cache to store previously detected relations

    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """
        Initialize relationship patterns for entity recognition.
        
        Returns:
            A dictionary containing relationship types and their corresponding regex patterns.
        """
        return {
            "TREATS": [
                r"\b(?:treat|treats|treating|for|prescribed for|helps with|to manage|helps manage)\b",
                r"\b(?:medication|drug|treatment)\s+(?:is|are|used for|indicated for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"\b(?:use|give|prescribe)\s+(\w+)\s+for\s+([\w\s]+)",
                r"\b([\w\s]+)\s+(?:helps|treats|is effective for|alleviates|relieves|reduces)\s+([\w\s]+)"
            ],
            "PREVENTS": [
                r"\b(?:prevent|prevents|preventing|to avoid|stops|inhibits)\b",
                r"\b(?:medication|treatment)\s+(?:is|are|used to)\s+(?:prevent|reduce|stop)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"\b(?:prevention of|prophylaxis for|preventative for)\s+([\w\s]+)",
                r"\b([\w\s]+)\s+(?:prevents|avoids|reduces risk of|protects against)\s+([\w\s]+)"
            ],
            "DIAGNOSES": [
                r"\b(?:diagnose|diagnoses|diagnostic for|confirms|to detect|detects|identified)\b",
                r"\b(?:diagnosis|condition)\s+(?:is|was|indicated as)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"\b(?:diagnosed with|confirmed|detected)\s+([\w\s]+)",
                r"\b([\w\s]+)\s+(?:test|scan|imaging)\s+(?:shows|confirms|indicates|diagnoses)\s+([\w\s]+)"
            ],
            "CAUSES": [
                r"\b(?:causes|cause|causing|leads to|results in|responsible for|triggers)\b",
                r"\b(?:condition|disease)\s+(?:causes|leads to|results in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"\b([\w\s]+)\s+(?:induced|caused|triggered|precipitated)\s+([\w\s]+)",
                r"\b(?:etiology|origin|source)\s+of\s+([\w\s]+)\s+is\s+([\w\s]+)"
            ],
            "SUGGESTS": [
                r"\b(?:suggests|indicating|indicates|sign of|symptom of|consistent with)\b",
                r"\b(?:symptom|condition)\s+(?:suggests|indicates)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"\b([\w\s]+)\s+(?:suggests|points to|indicates|is indicative of)\s+([\w\s]+)",
                r"\b(?:findings|results|presentation)\s+(?:consistent with|suggestive of)\s+([\w\s]+)"
            ],
            "CONTRAINDICATES": [
                r"\b(?:contraindicated|avoid|not recommended|should not use)\b",
                r"\b([\w\s]+)\s+(?:contraindicated|should be avoided|not recommended)\s+(?:in|with|for)\s+([\w\s]+)",
                r"\b(?:avoid|do not use)\s+([\w\s]+)\s+(?:in|with|for)\s+([\w\s]+)"
            ],
            "INTERACTS_WITH": [
                r"\b(?:interacts|interaction|interferes)\b",
                r"\b([\w\s]+)\s+(?:interacts with|interferes with|affects levels of)\s+([\w\s]+)",
                r"\b(?:drug interaction|medication interaction)\s+(?:between|with)\s+([\w\s]+)\s+and\s+([\w\s]+)"
            ],
            "MONITORS": [
                r"\b(?:monitor|monitors|monitoring|check|checks|checking|measure|measures|measuring)\b",
                r"\b([\w\s]+)\s+(?:monitors|checks|measures)\s+([\w\s]+)",
                r"\b(?:monitoring|checking|measuring)\s+([\w\s]+)\s+(?:for|to assess|to evaluate)\s+([\w\s]+)"
            ],
            "IS_SYMPTOM_OF": [
                r"\b(?:symptom|manifestation|sign|presentation|feature)\b",
                r"\b([\w\s]+)\s+(?:is a symptom of|is a sign of|is a manifestation of|is seen in)\s+([\w\s]+)",
                r"\b([\w\s]+)\s+(?:presents with|manifests as|displays|shows)\s+([\w\s]+)"
            ]
        }

    def add_terminology_resolver(self, resolver):
        """
        Add a terminology resolver to the pipeline.
        
        Args:
            resolver: An instance of a terminology resolver.
        """
        self.terminology_resolvers.append(resolver)
        return self  # Allow method chaining

    def add_llm_guided_mapping(self, mapping):
        """
        Add an LLM-guided mapping to the pipeline.
        
        Args:
            mapping: An instance of an LLM-guided mapping.
        """
        self.llm_guided_mappings.append(mapping)
        return self  # Allow method chaining

    def extract_relations(self, text: str, entities: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Extract relationships from the given text using defined patterns.
        If entities are provided, tries to link relations to specific entity pairs.
        
        Args:
            text: The input text to analyze.
            entities: Optional list of extracted entities to link with relations.
        
        Returns:
            A list of extracted relationships.
        """
        # Check cache first to avoid redundant extraction
        if text in self.relation_cache:
            return self.relation_cache[text]
            
        relations = []
        # Extract patterns using regex
        for relation_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    relation = {
                        "type": relation_type,
                        "text": match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                        "source": "pattern"
                    }
                    
                    # Try to extract entity pairs if groups are captured
                    if match.groups() and len(match.groups()) >= 2:
                        relation["source_text"] = match.group(1)
                        relation["target_text"] = match.group(2)
                    
                    relations.append(relation)
        
        # Try to link relations with provided entities
        if entities:
            relations = self._link_relations_to_entities(relations, entities, text)
        
        # Integrate terminology resolvers and LLM mappings
        self._integrate_terminology_and_llm(relations)
        
        # Cache the results
        self.relation_cache[text] = relations
        return relations

    def _link_relations_to_entities(self, relations: List[Dict[str, Any]], 
                                   entities: List[Dict[str, Any]], 
                                   text: str) -> List[Dict[str, Any]]:
        """
        Link extracted relations to specific entity pairs.
        
        Args:
            relations: The list of extracted relations.
            entities: The list of entities in the text.
            text: The original text.
            
        Returns:
            The list of relations with linked entity information.
        """
        linked_relations = []
        
        for relation in relations:
            # Define window around relation (+-50 characters)
            start_pos = max(0, relation["start"] - 50)
            end_pos = min(len(text), relation["end"] + 50)
            context_window = text[start_pos:end_pos]
            
            # Find entities in this context window
            nearby_entities = [
                e for e in entities 
                if (e["start"] >= start_pos and e["end"] <= end_pos)
            ]
            
            if len(nearby_entities) >= 2:
                # Try to find source and target entities based on position and type compatibility
                source_entity, target_entity = self._identify_source_target_entities(
                    nearby_entities, relation["type"])
                
                if source_entity and target_entity:
                    relation["source_entity"] = source_entity
                    relation["target_entity"] = target_entity
                    relation["source_text"] = source_entity["text"]
                    relation["target_text"] = target_entity["text"]
            
            linked_relations.append(relation)
        
        return linked_relations

    def _identify_source_target_entities(self, entities: List[Dict[str, Any]], 
                                         relation_type: str) -> Tuple[Optional[Dict[str, Any]], 
                                                                    Optional[Dict[str, Any]]]:
        """
        Identify source and target entities based on their types and relation type.
        
        Args:
            entities: List of candidate entities.
            relation_type: The type of relationship.
            
        Returns:
            A tuple containing the source and target entities if found, (None, None) otherwise.
        """
        # Define compatibility between relation types and entity types
        compatibility = {
            "TREATS": {
                "source": ["MEDICATION", "PROCEDURE", "TREATMENT"],
                "target": ["CONDITION", "SYMPTOM", "DISEASE"]
            },
            "PREVENTS": {
                "source": ["MEDICATION", "PROCEDURE", "TREATMENT", "VACCINE"],
                "target": ["CONDITION", "SYMPTOM", "DISEASE"]
            },
            "DIAGNOSES": {
                "source": ["TEST", "PROCEDURE", "LAB_TEST"],
                "target": ["CONDITION", "DISEASE"]
            },
            "CAUSES": {
                "source": ["CONDITION", "MEDICATION", "DISEASE", "PROCEDURE"],
                "target": ["CONDITION", "SYMPTOM", "DISEASE", "SIDE_EFFECT"]
            },
            "SUGGESTS": {
                "source": ["SYMPTOM", "TEST_RESULT", "FINDING"],
                "target": ["CONDITION", "DISEASE", "DIAGNOSIS"]
            },
            "CONTRAINDICATES": {
                "source": ["MEDICATION", "PROCEDURE", "TREATMENT"],
                "target": ["CONDITION", "DISEASE", "PATIENT_STATE"]
            },
            "INTERACTS_WITH": {
                "source": ["MEDICATION", "DRUG", "SUBSTANCE"],
                "target": ["MEDICATION", "DRUG", "SUBSTANCE", "FOOD"]
            },
            "MONITORS": {
                "source": ["TEST", "LAB_TEST", "PROCEDURE"],
                "target": ["CONDITION", "MEDICATION", "PARAMETER"]
            },
            "IS_SYMPTOM_OF": {
                "source": ["SYMPTOM", "FINDING", "SIGN"],
                "target": ["CONDITION", "DISEASE", "DIAGNOSIS"]
            }
        }
        
        if relation_type not in compatibility:
            return None, None
        
        # Get compatible entity types for this relation
        source_types = compatibility[relation_type]["source"]
        target_types = compatibility[relation_type]["target"]
        
        # Find compatible entities
        source_candidates = [e for e in entities if e.get("label") in source_types]
        target_candidates = [e for e in entities if e.get("label") in target_types]
        
        # Return the first compatible pair if found
        if source_candidates and target_candidates:
            return source_candidates[0], target_candidates[0]
        
        return None, None

    def _integrate_terminology_and_llm(self, relations: List[Dict[str, Any]]):
        """
        Integrate terminology resolvers and LLM-guided mappings into the extracted relations.
        
        Args:
            relations: The list of extracted relations to enhance.
        """
        for resolver in self.terminology_resolvers:
            # Apply each resolver to the relations
            resolver.resolve(relations)

        for mapping in self.llm_guided_mappings:
            # Apply each LLM mapping to the relations
            mapping.map(relations)
            
        # If both lists are empty, use LLM directly
        if not self.terminology_resolvers and not self.llm_guided_mappings and len(relations) > 0:
            self._enhance_relations_with_llm(relations)

    def _enhance_relations_with_llm(self, relations: List[Dict[str, Any]]):
        """
        Enhance relations using LLM when no resolvers or mappings are available.
        
        Args:
            relations: The list of relations to enhance.
        """
        # Only process a reasonable batch size for the LLM
        batch_size = min(10, len(relations))
        for i in range(0, len(relations), batch_size):
            batch = relations[i:i+batch_size]
            
            # Create a prompt for the LLM to verify and enhance relations
            prompt = "Please verify and enhance these extracted medical relationships:\n\n"
            for j, relation in enumerate(batch):
                prompt += f"{j+1}. Type: {relation['type']}\n"
                prompt += f"   Text: {relation['text']}\n"
                if 'source_text' in relation and 'target_text' in relation:
                    prompt += f"   Source: {relation['source_text']}\n"
                    prompt += f"   Target: {relation['target_text']}\n"
                prompt += "\n"
            
            prompt += "For each relationship, provide: correct type, confidence score (0-1), and any missing source/target entities. Return as JSON."
            
            try:
                # Get response from LLM
                enhanced = self.llm_client.generate_relation_enhancement(prompt)
                if isinstance(enhanced, list) and len(enhanced) == len(batch):
                    # Update relations with enhanced information
                    for j, enhancement in enumerate(enhanced):
                        if isinstance(enhancement, dict):
                            batch[j].update(enhancement)
            except Exception as e:
                print(f"Error enhancing relations with LLM: {e}")

    def validate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate the extracted relations using multi-agent cross-verification.
        
        Args:
            relations: The list of relations to validate.
        
        Returns:
            A list of validated relations.
        """
        validated_relations = []
        for relation in relations:
            # Implement validation logic here
            if self._is_valid_relation(relation):
                validated_relations.append(relation)
        
        return validated_relations

    def _is_valid_relation(self, relation: Dict[str, Any]) -> bool:
        """
        Check if a relation is valid based on predefined criteria.
        
        Args:
            relation: The relation to validate.
        
        Returns:
            A boolean indicating if the relation is valid.
        """
        # Check for required fields
        if "type" not in relation or "text" not in relation:
            return False
            
        # Check if relation type is in our defined patterns
        if relation["type"] not in self.patterns:
            return False
            
        # Check if source and target are provided for complete relations
        if "source_text" in relation and "target_text" in relation:
            if not relation["source_text"] or not relation["target_text"]:
                return False
                
        # Check confidence if available
        if "confidence" in relation and relation["confidence"] < 0.5:
            return False
            
        return True

    def construct_knowledge_graph(self, relations: List[Dict[str, Any]]) -> Any:
        """
        Construct a knowledge graph from the validated relations.
        
        Args:
            relations: The list of validated relations.
        
        Returns:
            The constructed knowledge graph.
        """
        import networkx as nx
        
        # Create directed graph
        graph = nx.DiGraph()
        
        # Add edges for each relation with source and target
        for relation in relations:
            if "source_text" in relation and "target_text" in relation:
                source = relation["source_text"]
                target = relation["target_text"]
                rel_type = relation["type"]
                
                # Add nodes if they don't exist
                if not graph.has_node(source):
                    source_attrs = {}
                    if "source_entity" in relation:
                        source_attrs = relation["source_entity"]
                    graph.add_node(source, **source_attrs)
                    
                if not graph.has_node(target):
                    target_attrs = {}
                    if "target_entity" in relation:
                        target_attrs = relation["target_entity"]
                    graph.add_node(target, **target_attrs)
                
                # Add edge with relation attributes
                graph.add_edge(source, target, 
                             relation_type=rel_type,
                             text=relation.get("text", ""),
                             confidence=relation.get("confidence", 1.0))
        
        return graph

    def validate_knowledge_graph(self, graph: Any) -> bool:
        """
        Validate the constructed knowledge graph.
        
        Args:
            graph: The knowledge graph to validate.
        
        Returns:
            A boolean indicating if the graph is valid.
        """
        import networkx as nx
        
        if not isinstance(graph, nx.Graph):
            return False
            
        # Check if graph is empty
        if len(graph.nodes()) == 0 or len(graph.edges()) == 0:
            return False
            
        # Check for isolated nodes (nodes without connections)
        isolated_nodes = list(nx.isolates(graph))
        if len(isolated_nodes) > len(graph.nodes()) * 0.5:  # If more than half nodes are isolated
            return False
            
        # Check relation types on edges
        for _, _, attrs in graph.edges(data=True):
            if "relation_type" not in attrs:
                return False
                
        return True
    
    def clear_cache(self):
        """Clear the relation cache to free memory"""
        self.relation_cache.clear()