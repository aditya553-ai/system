import numpy as np
from collections import Counter

def assign_confidence_scores(entity_mappings):
    """
    Assign confidence scores to entity mappings.
    
    Args:
        entity_mappings: Dict mapping entities to their resolved codes/terms
        
    Returns:
        Dict of confidence scores for each entity mapping
    """
    confidence_scores = {}
    
    for entity, mappings in entity_mappings.items():
        if not mappings:
            confidence_scores[entity] = 0.0
            continue
            
        # For single mappings or strings
        if isinstance(mappings, str) or not isinstance(mappings, list):
            mappings = [mappings]
        
        scores = []
        for mapping in mappings:
            # Calculate score based on mapping characteristics
            score = calculate_score(entity, mapping)
            scores.append(score)
        
        # Average score for the entity
        confidence_scores[entity] = np.mean(scores) if scores else 0.0
    
    return confidence_scores

def calculate_score(entity, mapping):
    """
    Calculate confidence score for a single entity mapping.
    
    Args:
        entity: The original entity text
        mapping: The mapped term/code
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    score = 0.5  # Base score
    
    # If mapping is a dict with metadata, extract relevant fields
    if isinstance(mapping, dict):
        # Check if there's an explicit confidence score
        if 'confidence' in mapping:
            return float(mapping['confidence'])
            
        # Extract the code/term for comparison
        mapped_term = mapping.get('term', mapping.get('code', ''))
        if 'score' in mapping:
            score = float(mapping['score'])
    else:
        mapped_term = str(mapping)
    
    # String similarity between entity and mapped term (if it's a term)
    if isinstance(mapped_term, str) and mapped_term:
        similarity = string_similarity(entity.lower(), mapped_term.lower())
        score += 0.3 * similarity
    
    # Assess code validity if it's a standardized code
    if isinstance(mapped_term, str) and is_valid_medical_code(mapped_term):
        score += 0.1
        
    # Cap score at 1.0
    return min(1.0, score)

def string_similarity(s1, s2):
    """Simple string similarity metric"""
    # Count common words
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    common_words = words1.intersection(words2)
    
    # Calculate Jaccard similarity
    if not words1 and not words2:
        return 1.0
    return len(common_words) / max(1, len(words1.union(words2)))

def is_valid_medical_code(code):
    """Check if the string has a valid medical code format"""
    # Basic checks for common code formats
    
    # ICD-10 format (e.g., E11.9)
    if len(code) >= 3 and code[0].isalpha() and code[1:3].isdigit():
        return True
    
    # CPT format (5 digits)
    if len(code) == 5 and code.isdigit():
        return True
    
    # RxNorm format (mostly numeric)
    if code.startswith('RxNorm:') or code.isdigit():
        return True
    
    # LOINC format (digits-digits)
    if '-' in code and all(part.isdigit() for part in code.split('-')):
        return True
    
    # SNOMED CT format (digits)
    if code.isdigit() and len(code) > 5:
        return True
    
    return False

def flag_low_confidence(confidence_scores, threshold=0.5):
    """
    Flag entities with confidence scores below threshold.
    
    Args:
        confidence_scores: Dict of confidence scores by entity
        threshold: Minimum acceptable confidence score
        
    Returns:
        List of flagged entities with their scores
    """
    flagged_entities = []
    
    for entity, score in confidence_scores.items():
        if score < threshold:
            flagged_entities.append({
                'entity': entity,
                'confidence': score,
                'reason': 'Low confidence score'
            })
    
    return flagged_entities

def get_best_mapping(entity_mappings, confidence_scores):
    """
    Get the best mapping for each entity based on confidence scores.
    
    Args:
        entity_mappings: Dict of entities to their possible mappings
        confidence_scores: Dict of confidence scores for mappings
        
    Returns:
        Dict of entities to their best mappings
    """
    best_mappings = {}
    
    for entity, mappings in entity_mappings.items():
        if not mappings:
            continue
            
        # For single mappings or strings
        if isinstance(mappings, str) or not isinstance(mappings, list):
            best_mappings[entity] = mappings
            continue
            
        # If we have multiple mappings, choose the one with highest confidence
        # This is a simplified version - in reality you'd calculate confidence per mapping
        best_mapping = mappings[0]
        if len(mappings) > 1:
            # Sort by estimated confidence
            scores = [calculate_score(entity, m) for m in mappings]
            best_idx = np.argmax(scores)
            best_mapping = mappings[best_idx]
            
        best_mappings[entity] = best_mapping
        
    return best_mappings