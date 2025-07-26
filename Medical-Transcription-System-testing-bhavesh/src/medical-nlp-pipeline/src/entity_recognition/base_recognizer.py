from typing import Dict, List, Any, Optional, Tuple
import re
import logging
from collections import defaultdict

class BaseRecognizer:
    """
    Base class for entity recognizers, defining common methods and properties for all recognizers.
    """

    def __init__(self):
        self.entity_types = []
        self.logger = logging.getLogger(__name__)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from the given text.

        Args:
            text: The input text from which to extract entities.

        Returns:
            A list of extracted entities, each represented as a dictionary.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def validate_entity(self, entity: Dict[str, Any]) -> bool:
        """
        Validate the extracted entity.

        Args:
            entity: The entity to validate.

        Returns:
            True if the entity is valid, False otherwise.
        """
        # Basic validation - entity should have text and a valid type
        if not entity.get("text"):
            return False
            
        if entity.get("label") not in self.entity_types and self.entity_types:
            return False
            
        # Additional validation - entity should have start and end positions
        if "start" not in entity or "end" not in entity:
            return False
            
        # Entity text should not be just whitespace or punctuation
        if re.match(r'^\s*[\.,;:\s]*\s*$', entity.get("text", "")):
            return False
            
        return True

    def process_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Process a batch of texts to extract entities from each.
        
        Args:
            texts: List of text strings to process
            
        Returns:
            List of lists of entities, one list per input text
        """
        results = []
        for text in texts:
            try:
                entities = self.extract_entities(text)
                results.append(entities)
            except Exception as e:
                self.logger.error(f"Error processing text: {e}")
                results.append([])
        return results

    def filter_entities(self, entities: List[Dict[str, Any]], 
                       types: Optional[List[str]] = None, 
                       min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """
        Filter entities by type and/or confidence score.
        
        Args:
            entities: List of entities to filter
            types: Optional list of entity types to keep
            min_confidence: Minimum confidence score threshold
            
        Returns:
            Filtered list of entities
        """
        if not entities:
            return []
            
        filtered = []
        for entity in entities:
            # Filter by type if specified
            if types and entity.get("label") not in types:
                continue
                
            # Filter by confidence if present
            if "confidence" in entity and entity["confidence"] < min_confidence:
                continue
                
            # Entity passed all filters
            filtered.append(entity)
            
        return filtered

    def merge_overlapping_entities(self, entities: List[Dict[str, Any]], 
                                  priority_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Merge overlapping entities based on priority or confidence.
        
        Args:
            entities: List of entities to merge
            priority_types: List of entity types in priority order
            
        Returns:
            List of entities with overlaps resolved
        """
        if not entities or len(entities) <= 1:
            return entities
            
        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda e: (e.get("start", 0), e.get("end", 0)))
        
        # Resolve overlaps
        merged = []
        current = sorted_entities[0]
        
        for next_entity in sorted_entities[1:]:
            # Check for overlap
            if next_entity.get("start") <= current.get("end"):
                # Entities overlap, decide which to keep
                keep_current = True
                
                # If priority types are provided, use them to decide
                if priority_types:
                    current_priority = priority_types.index(current.get("label")) if current.get("label") in priority_types else len(priority_types)
                    next_priority = priority_types.index(next_entity.get("label")) if next_entity.get("label") in priority_types else len(priority_types)
                    
                    if next_priority < current_priority:
                        keep_current = False
                # Otherwise decide based on confidence
                elif "confidence" in next_entity and "confidence" in current:
                    if next_entity["confidence"] > current["confidence"]:
                        keep_current = False
                
                # Replace current entity if needed
                if not keep_current:
                    current = next_entity
            else:
                # No overlap, add current to results and move to next
                merged.append(current)
                current = next_entity
                
        # Add the last entity
        merged.append(current)
        
        return merged
        
    def post_process_entities(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """
        Perform post-processing on extracted entities.
        
        Args:
            entities: List of entities to post-process
            text: Original text for context
            
        Returns:
            Post-processed list of entities
        """
        if not entities:
            return []
            
        processed = []
        
        for entity in entities:
            # Validate the entity
            if not self.validate_entity(entity):
                continue
                
            # Ensure entity text matches the text at its position
            if "start" in entity and "end" in entity:
                span_text = text[entity["start"]:entity["end"]]
                if span_text.strip() != entity["text"].strip():
                    # Try to fix misaligned spans
                    entity["text"] = span_text
            
            processed.append(entity)
            
        return processed
        
    def group_entities_by_type(self, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group entities by their type/label.
        
        Args:
            entities: List of entities to group
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        groups = defaultdict(list)
        
        for entity in entities:
            if "label" in entity:
                groups[entity["label"]].append(entity)
                
        return dict(groups)