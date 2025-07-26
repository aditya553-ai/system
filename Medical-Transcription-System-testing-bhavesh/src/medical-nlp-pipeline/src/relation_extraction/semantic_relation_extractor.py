from typing import Dict, List, Any, Optional, Tuple
import spacy
from spacy.tokens import Doc
import networkx as nx
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import numpy as np
from entity_recognition.medical_entity_recognizer import MedicalEntityRecognizer
from llm.llm_client import LLMClient
from model_manager import ModelManager
from .relation_patterns import RelationPatterns
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



class SemanticRelationExtractor:
    """
    Extract semantic relationships between medical entities in text
    using rule-based, LLM-guided approaches, and knowledge graph validation.
    """
    
    def __init__(self,model_manager_instance: ModelManager, nlp=None, use_llm=True):
        """
        Initialize the semantic relation extractor.
        
        Args:
            nlp: Optional spaCy NLP model for processing text
            use_llm: Whether to use LLM for enhanced relation extraction
        """
        # Initialize NLP if not provided
        if nlp is None:
            self.model_manager = model_manager_instance
            self.nlp = MedicalEntityRecognizer(model_manager_instance=self.model_manager)
        else:
            self.nlp = nlp
            
        self.use_llm = use_llm
        self.llm_client = LLMClient(model_manager_instance=self.model_manager) if use_llm else None
        self.llm_model = None
        self.llm_tokenizer = None
        self.relation_patterns = RelationPatterns(model_manager_instance=self.model_manager)
        self.knowledge_graph = nx.DiGraph()
        
        self.relation_types = [
            "TREATS", "PREVENTS", "DIAGNOSES", "CAUSES", "SUGGESTS",
            "CONTRAINDICATES", "INTERACTS_WITH", "MONITORS", "IS_SYMPTOM_OF"
        ]
        
        if self.use_llm:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM for enhanced relation extraction"""
        try:
            model_name = "bvanaken/clinical-assertion-negation-bert"
            print(f"Loading relation extraction model: {model_name}")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            print("Relation extraction model loaded successfully")
        except Exception as e:
            print(f"Error loading relation extraction model: {e}")
            self.use_llm = False

    def _extract_relations_from_context(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relations based on contextual understanding of the text
        
        Args:
            text: Original text
            entities: List of entities
            
        Returns:
            List of relations with contextual modifiers
        """
        relations = []
        
        # First categorize entities by type for easier processing
        entity_map = {}
        for entity in entities:
            entity_type = entity.get('entity_type', 'UNKNOWN')
            if entity_type not in entity_map:
                entity_map[entity_type] = []
            entity_map[entity_type].append(entity)
        
        # Process medication entities to determine if they TREAT or CAUSE symptoms/conditions
        if 'MEDICATION' in entity_map:
            for med_entity in entity_map['MEDICATION']:
                # Check if this medication has modifiers
                has_negation = False
                if 'modifiers' in med_entity and 'negation' in med_entity['modifiers'] and med_entity['modifiers']['negation']:
                    has_negation = True
                    
                # Get context around the medication (either from entity or extract it)
                if 'context' not in med_entity:
                    start = max(0, med_entity['start'] - 100)
                    end = min(len(text), med_entity['end'] + 100)
                    context = text[start:end].lower()
                else:
                    context = med_entity['context'].lower()
                    
                # Look for treatment patterns
                treatment_indicators = ["prescribed", "for", "helps with", "treats", "to treat", 
                                    "for relief", "to help", "will help", "to reduce"]
                
                # Look for causal patterns
                causal_indicators = ["avoid", "reduce", "stop", "triggers", "causes", 
                                "makes worse", "aggravates", "don't", "do not"]
                
                # Determine relation type based on context
                is_treatment = any(indicator in context for indicator in treatment_indicators)
                is_causal = any(indicator in context for indicator in causal_indicators) or has_negation
                
                # If both are present, the more specific wins
                if is_treatment and is_causal:
                    # Check which indicator is closer to the medication mention
                    med_pos = context.find(med_entity['text'].lower())
                    closest_treatment_distance = min(
                        abs(med_pos - context.find(indicator)) 
                        for indicator in treatment_indicators if indicator in context
                    )
                    closest_causal_distance = min(
                        abs(med_pos - context.find(indicator)) 
                        for indicator in causal_indicators if indicator in context
                    )
                    
                    # Use the closest indicator's meaning
                    if closest_treatment_distance <= closest_causal_distance and not has_negation:
                        is_treatment = True
                        is_causal = False
                    else:
                        is_treatment = False
                        is_causal = True
                
                # Add relations to conditions
                if 'CONDITION' in entity_map:
                    for cond_entity in entity_map['CONDITION']:
                        if is_treatment and not is_causal:
                            relations.append({
                                'source': med_entity['text'],
                                'source_id': med_entity.get('id', ''),
                                'target': cond_entity['text'],
                                'target_id': cond_entity.get('id', ''),
                                'relation_type': 'TREATS',
                                'confidence': 0.8,
                                'context': context[:100]  # Add a snippet of context
                            })
                        elif is_causal:
                            relations.append({
                                'source': med_entity['text'],
                                'source_id': med_entity.get('id', ''),
                                'target': cond_entity['text'],
                                'target_id': cond_entity.get('id', ''),
                                'relation_type': 'TRIGGERS',  # or CAUSES or AGGRAVATES
                                'confidence': 0.7,
                                'context': context[:100]  # Add a snippet of context
                            })
                
                # Add relations to symptoms
                if 'SYMPTOM' in entity_map:
                    for symptom_entity in entity_map['SYMPTOM']:
                        if is_treatment and not is_causal:
                            relations.append({
                                'source': med_entity['text'],
                                'source_id': med_entity.get('id', ''),
                                'target': symptom_entity['text'],
                                'target_id': symptom_entity.get('id', ''),
                                'relation_type': 'RELIEVES',
                                'confidence': 0.8,
                                'context': context[:100]  # Add a snippet of context
                            })
                        elif is_causal:
                            relations.append({
                                'source': med_entity['text'],
                                'source_id': med_entity.get('id', ''),
                                'target': symptom_entity['text'],
                                'target_id': symptom_entity.get('id', ''),
                                'relation_type': 'TRIGGERS',  # or CAUSES or AGGRAVATES
                                'confidence': 0.7,
                                'context': context[:100]  # Add a snippet of context
                            })
        
        # Add relations between symptoms and conditions
        if 'SYMPTOM' in entity_map and 'CONDITION' in entity_map:
            for symptom_entity in entity_map['SYMPTOM']:
                for cond_entity in entity_map['CONDITION']:
                    relations.append({
                        'source': symptom_entity['text'],
                        'source_id': symptom_entity.get('id', ''),
                        'target': cond_entity['text'],
                        'target_id': cond_entity.get('id', ''),
                        'relation_type': 'INDICATES',
                        'confidence': 0.6,
                        'method': 'context'
                    })
        
        return relations
    
    def extract_relations(self, entities, text=None):
        """
        Extract relations between medical entities
        
        Args:
            entities: List of extracted and normalized entities
            text: Optional original text for context (can be None when working with just entities)
            
        Returns:
            List of relations between entities
        """
        relations = []
        
        if not entities:
            return relations
        
        # Get the full text from transcript if needed
        if isinstance(text, dict) and "speaker_turns" in text:
            full_text = ""
            for turn in text["speaker_turns"]:
                if "text" in turn:
                    full_text += turn["text"] + " "
            text = full_text.strip()
        
        # Use entity-based rule extraction (works without original text)
        rule_relations = self._extract_relations_from_rules(entities)
        relations.extend(rule_relations)
        
        # If text is provided, use pattern and dependency-based extraction
        if text and isinstance(text, str):
            try:
                if hasattr(self, 'relation_patterns'):
                    pattern_relations = self.relation_patterns.extract_relations(text, entities)
                    relations.extend(pattern_relations)
            except Exception as e:
                print(f"Error in pattern relation extraction: {e}")
            
            try:
                dependency_relations = self._extract_relations_from_dependencies(text, entities)
                relations.extend(dependency_relations)
            except Exception as e:
                print(f"Error in dependency relation extraction: {e}")
                
            # Apply contextual relation extraction - critical for understanding meaning
            try:
                context_relations = self._extract_relations_from_context(text, entities)
                relations.extend(context_relations)
            except Exception as e:
                print(f"Error in contextual relation extraction: {e}")
        
        # If enabled and we have text, use LLM to extract additional relations
        if self.use_llm and text and isinstance(text, str):
            try:
                llm_relations = self._extract_relations_with_llm(text, entities)
                relations.extend(llm_relations)
            except Exception as e:
                print(f"Error in LLM relation extraction: {e}")
            
        # Filter out incorrect relations based on medical knowledge
        relations = self._filter_relations(relations, entities, text)
            
        # Remove duplicates
        unique_relations = []
        relation_tuples = set()
        
        for rel in relations:
            rel_tuple = (rel.get('source', ''), rel.get('target', ''), rel.get('relation_type', ''))
            if rel_tuple not in relation_tuples:
                relation_tuples.add(rel_tuple)
                unique_relations.append(rel)
                
        return unique_relations

    def _filter_relations(self, relations, entities, text):
        """Filter out incorrect or nonsensical relations"""
        # Map entities by text for quick lookup
        entity_map = {entity.get('text', ''): entity for entity in entities}
        
        filtered_relations = []
        for relation in relations:
            source_text = relation.get('source', '')
            target_text = relation.get('target', '')
            relation_type = relation.get('relation_type', '')
            
            # Skip if source or target doesn't exist
            if source_text not in entity_map or target_text not in entity_map:
                continue
                
            source_entity = entity_map[source_text]
            target_entity = entity_map[target_text]
            
            # Check for nonsensical relations
            source_type = source_entity.get('entity_type', '')
            target_type = target_entity.get('entity_type', '')
            
            # Filter out incorrect medication relations
            if source_type == 'MEDICATION':
                if relation_type == 'TREATS' or relation_type == 'RELIEVES':
                    # Check for negation in context
                    if text:
                        # Look for specific phrases around source and target
                        source_pos = text.find(source_text)
                        target_pos = text.find(target_text)
                        
                        if source_pos >= 0 and target_pos >= 0:
                            # Look at context between the entities 
                            start = min(source_pos, target_pos)
                            end = max(source_pos + len(source_text), target_pos + len(target_text))
                            context = text[max(0, start-30):min(len(text), end+30)].lower()
                            
                            # Check for negation words
                            negation_terms = ["avoid", "reduce", "stop", "don't", "do not", "without",
                                            "discontinue", "no more", "causes", "triggers"]
                                            
                            if any(term in context for term in negation_terms):
                                # Change relation type to TRIGGERS instead of TREATS
                                relation['relation_type'] = 'TRIGGERS'
                                relation['confidence'] = 0.8
                                
            # Special case for caffeine
            if source_text.lower() == "caffeine" and relation_type == 'TREATS' and 'headache' in target_text.lower():
                # Find out more context
                if text and "reduce caffeine" in text.lower():
                    relation['relation_type'] = 'TRIGGERS'
                    relation['confidence'] = 0.9
                    relation['note'] = "Changed from TREATS to TRIGGERS due to 'reduce caffeine' context"
                
            filtered_relations.append(relation)
            
        return filtered_relations

    def _extract_relations_from_rules(self, entities):
        """Extract relations based on entity types and rules"""
        relations = []
        
        # Create entity lookup by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.get('entity_type')
            if entity_type:
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(entity)
        
        # Apply medical domain rules
        
        # Rule: Medication treats Condition
        if 'MEDICATION' in entities_by_type and 'CONDITION' in entities_by_type:
            for med_entity in entities_by_type['MEDICATION']:
                for cond_entity in entities_by_type['CONDITION']:
                    relations.append({
                        'source': med_entity['text'],
                        'source_id': med_entity.get('id', ''),
                        'target': cond_entity['text'],
                        'target_id': cond_entity.get('id', ''),
                        'relation_type': 'TREATS',
                        'confidence': 0.6
                    })
        
        # Rule: Test diagnoses Condition
        if 'TEST' in entities_by_type and 'CONDITION' in entities_by_type:
            for test_entity in entities_by_type['TEST']:
                for cond_entity in entities_by_type['CONDITION']:
                    relations.append({
                        'source': test_entity['text'],
                        'source_id': test_entity.get('id', ''),
                        'target': cond_entity['text'], 
                        'target_id': cond_entity.get('id', ''),
                        'relation_type': 'DIAGNOSES',
                        'confidence': 0.6
                    })
        
        # Rule: Condition affects Anatomy
        if 'CONDITION' in entities_by_type and 'ANATOMY' in entities_by_type:
            for cond_entity in entities_by_type['CONDITION']:
                for anat_entity in entities_by_type['ANATOMY']:
                    relations.append({
                        'source': cond_entity['text'],
                        'source_id': cond_entity.get('id', ''),
                        'target': anat_entity['text'],
                        'target_id': anat_entity.get('id', ''),
                        'relation_type': 'AFFECTS',
                        'confidence': 0.7
                    })
        
        return relations
    
    def _extract_dependency_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relations using dependency parsing.
        
        Args:
            text: The text to analyze
            entities: List of extracted entities
            
        Returns:
            List of relations extracted through dependency parsing
        """
        relations = []
        
        try:
            # Process with spaCy to get dependency information
            doc = self.nlp.nlp(text)
            
            # Create entity span mapping by character position
            entity_spans = {}
            for entity in entities:
                for i in range(entity["start"], entity["end"]):
                    entity_spans[i] = entity
            
            # Look for dependency patterns that suggest relations
            for token in doc:
                # Check for verb-mediated relations
                if token.pos_ == "VERB":
                    # Find subject and object connected to this verb
                    subject = None
                    direct_object = None
                    
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subject = child
                        elif child.dep_ in ["dobj", "pobj"]:
                            direct_object = child
                    
                    # If we found both subject and object, check if they map to entities
                    if subject and direct_object:
                        subj_span = doc[subject.left_edge.i:subject.right_edge.i+1]
                        obj_span = doc[direct_object.left_edge.i:direct_object.right_edge.i+1]
                        
                        # Check if these spans overlap with any entities
                        source_entity = None
                        target_entity = None
                        
                        for i in range(subj_span.start_char, subj_span.end_char):
                            if i in entity_spans:
                                source_entity = entity_spans[i]
                                break
                                
                        for i in range(obj_span.start_char, obj_span.end_char):
                            if i in entity_spans:
                                target_entity = entity_spans[i]
                                break
                        
                        # If both entities are found, create a relation
                        if source_entity and target_entity and source_entity != target_entity:
                            verb_text = token.text.lower()
                            
                            # Determine relation type based on verb
                            relation_type = self._determine_relation_type_from_verb(verb_text)
                            
                            if relation_type:
                                relations.append({
                                    "type": relation_type,
                                    "text": f"{subj_span.text} {token.text} {obj_span.text}",
                                    "start": min(subj_span.start_char, obj_span.start_char),
                                    "end": max(subj_span.end_char, obj_span.end_char),
                                    "source_text": source_entity["text"],
                                    "target_text": target_entity["text"],
                                    "source_entity": source_entity,
                                    "target_entity": target_entity,
                                    "confidence": 0.7,  # Lower confidence for dependency-based
                                    "source": "dependency"
                                })
        except Exception as e:
            print(f"Error in dependency relation extraction: {e}")
        
        return relations
    
    def _determine_relation_type_from_verb(self, verb: str) -> Optional[str]:
        """
        Determine relation type based on verb.
        
        Args:
            verb: The verb to analyze
            
        Returns:
            Relation type if matched, None otherwise
        """
        # Map common verbs to relation types
        verb_mappings = {
            "treat": "TREATS",
            "prevent": "PREVENTS",
            "diagnose": "DIAGNOSES",
            "cause": "CAUSES",
            "suggest": "SUGGESTS",
            "indicate": "SUGGESTS",
            "contraindicate": "CONTRAINDICATES",
            "interact": "INTERACTS_WITH",
            "monitor": "MONITORS",
            "check": "MONITORS",
            "measure": "MONITORS",
            "symptom": "IS_SYMPTOM_OF"
        }
        
        # Check for exact matches
        if verb in verb_mappings:
            return verb_mappings[verb]
        
        # Check for partial matches (e.g., "treating" -> "treat")
        for v, rel_type in verb_mappings.items():
            if verb.startswith(v):
                return rel_type
        
        return None

    def _extract_relations_from_dependencies(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relations between entities using dependency parsing.
        
        Args:
            text: The original text
            entities: List of entities to find relations between
            
        Returns:
            List of extracted relations
        """
        relations = []
        
        try:
            # Create a dedicated spaCy model for dependency parsing
            # Don't reuse self.nlp which is a MedicalEntityRecognizer
            import spacy
            try:
                # Load a fresh spaCy model for this operation
                nlp = spacy.load("en_core_web_sm")
                print("Loaded fresh spaCy model for dependency parsing")
            except Exception as e:
                print(f"Error loading spaCy model: {e}")
                # Try an alternative loading method
                import en_core_web_sm
                nlp = en_core_web_sm.load()
                print("Loaded spaCy model via module import")
            
            # Process the text with our fresh spaCy instance
            doc = nlp(text)
            
            # Create a map of entity spans to entity objects
            entity_spans = {}
            for entity in entities:
                if 'start' in entity and 'end' in entity:
                    start = entity['start']
                    end = entity['end']
                    entity_spans[(start, end)] = entity
            
            # Look for relations based on dependency patterns
            for sent in doc.sents:
                # Find verbs that might indicate relations
                for token in sent:
                    if token.pos_ == "VERB":
                        # Check subject-verb-object patterns
                        subject = None
                        objects = []
                        
                        # Find subject
                        for child in token.children:
                            if child.dep_ in ["nsubj", "nsubjpass"]:
                                subject = child
                                break
                        
                        # Find objects
                        for child in token.children:
                            if child.dep_ in ["dobj", "pobj", "attr"]:
                                objects.append(child)
                        
                        # Skip if no subject or objects found
                        if not subject or not objects:
                            continue
                        
                        # Check if subject and any object correspond to entities
                        subject_entity = self._find_entity_for_token(subject, entity_spans)
                        
                        for obj in objects:
                            obj_entity = self._find_entity_for_token(obj, entity_spans)
                            
                            if subject_entity and obj_entity:
                                # Create a relation
                                relation_type = self._determine_relation_type(token.text, 
                                                                            subject_entity.get('entity_type', ''),
                                                                            obj_entity.get('entity_type', ''))
                                
                                if relation_type:
                                    relations.append({
                                        'source': subject_entity['text'],
                                        'source_id': subject_entity.get('id', ''),
                                        'target': obj_entity['text'],
                                        'target_id': obj_entity.get('id', ''),
                                        'relation_type': relation_type,
                                        'confidence': 0.7,
                                        'method': 'dependency'
                                    })
            
            return relations
        except Exception as e:
            import traceback
            print(f"Error in dependency relation extraction: {e}")
            traceback.print_exc()
            return []
            
    def _find_entity_for_token(self, token, entity_spans):
        """Find which entity a token belongs to"""
        token_start = token.idx
        token_end = token_start + len(token.text)
        
        # Check if token is within any entity span
        for (start, end), entity in entity_spans.items():
            if token_start >= start and token_end <= end:
                return entity
                
        return None
        
    def _determine_relation_type(self, verb, source_type, target_type):
        """Determine the relation type based on verb and entity types"""
        verb = verb.lower()
        
        # Medical domain-specific relation mappings
        if source_type == 'MEDICATION' and target_type == 'CONDITION':
            if verb in ['treat', 'help', 'relieve', 'cure', 'manage']:
                return 'TREATS'
            elif verb in ['cause', 'induce', 'trigger']:
                return 'CAUSES'
        
        elif source_type == 'TEST' and target_type == 'CONDITION':
            if verb in ['diagnose', 'detect', 'identify', 'confirm']:
                return 'DIAGNOSES'
            
        elif source_type == 'SYMPTOM' and target_type == 'CONDITION':
            if verb in ['indicate', 'suggest', 'point to']:
                return 'INDICATES'
                
        elif source_type == 'CONDITION' and target_type == 'ANATOMY':
            if verb in ['affect', 'impact', 'involve']:
                return 'AFFECTS'
        
        # Default relation types based just on verb
        if verb in ['treat', 'cure', 'heal', 'manage']:
            return 'TREATS'
        elif verb in ['cause', 'trigger', 'induce', 'lead to']:
            return 'CAUSES'
        elif verb in ['indicate', 'suggest', 'imply']:
            return 'INDICATES'
        elif verb in ['diagnose', 'detect', 'identify']:
            return 'DIAGNOSES'
        elif verb in ['show', 'reveal', 'demonstrate']:
            return 'REVEALS'
                
        # Default relation when we can't determine a specific type
        return 'RELATED_TO'
    
    def _extract_llm_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relations using LLM.
        
        Args:
            text: The text to analyze
            entities: List of extracted entities
            
        Returns:
            List of relations extracted by the LLM
        """
        if not self.llm_client:
            return []
            
        relations = []
        
        try:
            # Create a prompt for the LLM
            entity_text = ", ".join([f"{e['text']} ({e['label']})" for e in entities])
            prompt = f"""
            Extract medical relationships between the following entities in this text:
            
            Text: "{text}"
            
            Entities: {entity_text}
            
            For each relation, specify the source entity, target entity, and relation type 
            from these types: TREATS, PREVENTS, DIAGNOSES, CAUSES, SUGGESTS, CONTRAINDICATES, 
            INTERACTS_WITH, MONITORS, IS_SYMPTOM_OF.
            
            Return results as JSON with an array of relations.
            """
            
            # Get LLM response
            response = self.llm_client.generate_relation_extraction(prompt)
            
            # Parse response
            if isinstance(response, dict) and "relations" in response:
                llm_relations = response["relations"]
                
                # Convert to our relation format
                for rel in llm_relations:
                    if "source" in rel and "target" in rel and "type" in rel:
                        # Find matching entities
                        source_entity = next((e for e in entities if e["text"].lower() == rel["source"].lower()), None)
                        target_entity = next((e for e in entities if e["text"].lower() == rel["target"].lower()), None)
                        
                        if source_entity and target_entity:
                            relations.append({
                                "type": rel["type"],
                                "text": f"{source_entity['text']} {rel.get('verb', 'related to')} {target_entity['text']}",
                                "source_text": source_entity["text"],
                                "target_text": target_entity["text"],
                                "source_entity": source_entity,
                                "target_entity": target_entity,
                                "confidence": rel.get("confidence", 0.8),
                                "source": "llm"
                            })
            
        except Exception as e:
            print(f"Error in LLM relation extraction: {e}")
        
        return relations

    def _extract_relations_with_llm(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use LLM to extract relations between medical entities.
        
        Args:
            text: The original text
            entities: List of entities to find relations between
            
        Returns:
            List of extracted relations
        """
        # Skip if no LLM client or not enough entities
        if not hasattr(self, 'llm_client') or self.llm_client is None or len(entities) < 2:
            return []
            
        try:
            # First gather existing relations to pass to the enhancement function
            existing_relations = []
            
            # If we have relation_patterns, get those relations first
            if hasattr(self, 'relation_patterns') and self.relation_patterns:
                try:
                    pattern_relations = self.relation_patterns.extract_relations(text, entities)
                    existing_relations.extend(pattern_relations)
                except Exception as e:
                    print(f"Error getting pattern relations: {e}")
            
            # Use the LLM to enhance relations
            enhanced_relations = self.llm_client.generate_relation_enhancement(entities, existing_relations)
            
            # Format the relations properly
            formatted_relations = []
            for rel in enhanced_relations:
                # Skip incomplete relations
                if not isinstance(rel, dict) or 'source' not in rel or 'target' not in rel:
                    continue
                    
                # Extract the source and target entities
                source_text = rel.get('source', '')
                target_text = rel.get('target', '')
                relation_type = rel.get('relation_type', 'RELATED_TO')
                
                # Find matching entities in our entity list
                source_entity = None
                target_entity = None
                
                for entity in entities:
                    entity_text = entity.get('text', '').lower()
                    # Match on text or preferred term
                    entity_terms = [entity_text]
                    if 'preferred_term' in entity:
                        entity_terms.append(entity.get('preferred_term', '').lower())
                        
                    if any(source_text.lower() in term or term in source_text.lower() for term in entity_terms):
                        source_entity = entity
                    if any(target_text.lower() in term or term in target_text.lower() for term in entity_terms):
                        target_entity = entity
                
                # Create relation if we found both entities
                if source_entity and target_entity:
                    relation = {
                        'source': source_entity.get('text', ''),
                        'source_id': source_entity.get('id', ''),
                        'target': target_entity.get('text', ''),
                        'target_id': target_entity.get('id', ''),
                        'relation_type': relation_type,
                        'confidence': rel.get('confidence', 0.7),
                        'method': 'llm'
                    }
                    formatted_relations.append(relation)
                    
            return formatted_relations
        except Exception as e:
            print(f"Error in LLM relation extraction: {e}")
            return []
    
    def _deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate relations based on source/target pairs.
        
        Args:
            relations: List of relations to deduplicate
            
        Returns:
            Deduplicated list of relations
        """
        if not relations:
            return []
            
        # Track seen relation pairs
        seen_pairs = set()
        unique_relations = []
        
        for relation in relations:
            if "source_text" in relation and "target_text" in relation:
                # Create a key for this source-target-type combination
                pair_key = (
                    relation["source_text"].lower(),
                    relation["target_text"].lower(),
                    relation["type"]
                )
                
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    unique_relations.append(relation)
                else:
                    # If duplicate, keep the one with higher confidence
                    existing_idx = next(
                        (i for i, r in enumerate(unique_relations) 
                         if (r["source_text"].lower() == pair_key[0] and
                             r["target_text"].lower() == pair_key[1] and
                             r["type"] == pair_key[2])),
                        None
                    )
                    
                    if existing_idx is not None and relation.get("confidence", 0) > unique_relations[existing_idx].get("confidence", 0):
                        unique_relations[existing_idx] = relation
            else:
                # If missing source/target, still include the relation
                unique_relations.append(relation)
        
        return unique_relations
    
    def _update_knowledge_graph(self, relations: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> None:
        """
        Update the knowledge graph with new relations.
        
        Args:
            relations: List of relations to add
            entities: List of entities
        """
        # Create entity lookup by text
        entity_lookup = {e["text"]: e for e in entities}
        
        # Add relations to knowledge graph
        for relation in relations:
            if "source_text" in relation and "target_text" in relation:
                source = relation["source_text"]
                target = relation["target_text"]
                
                # Add nodes if needed
                if not self.knowledge_graph.has_node(source):
                    source_attrs = entity_lookup.get(source, {})
                    self.knowledge_graph.add_node(source, **source_attrs)
                
                if not self.knowledge_graph.has_node(target):
                    target_attrs = entity_lookup.get(target, {})
                    self.knowledge_graph.add_node(target, **target_attrs)
                
                # Add edge with relation information
                self.knowledge_graph.add_edge(
                    source, target,
                    relation=relation["type"],
                    text=relation["text"],
                    confidence=relation.get("confidence", 1.0)
                )
    
    def _determine_relation(self, text: str, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Determine the relationship between two entities.
        
        Args:
            text: The context text
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Relation dictionary if a relation exists, None otherwise
        """
        # Get spans between entities
        if entity1["start"] < entity2["start"]:
            start_entity = entity1
            end_entity = entity2
            span_start = entity1["end"]
            span_end = entity2["start"]
        else:
            start_entity = entity2
            end_entity = entity1
            span_start = entity2["end"]
            span_end = entity1["start"]
        
        # Extract the text between entities
        between_text = text[span_start:span_end]
        
        # Skip if entities are too far apart (> 100 chars)
        if len(between_text) > 100:
            return None
        
        # Check for relation patterns in the connecting text
        for relation_type, patterns in self.relation_patterns.patterns.items():
            for pattern in patterns:
                if re.search(pattern, between_text, re.IGNORECASE):
                    # Determine direction based on entity types and relation type
                    if self._is_valid_relation_direction(start_entity, end_entity, relation_type):
                        return {
                            "type": relation_type,
                            "text": between_text,
                            "source_text": start_entity["text"],
                            "target_text": end_entity["text"],
                            "source_entity": start_entity,
                            "target_entity": end_entity,
                            "confidence": 0.8,
                            "source": "pattern"
                        }
                    else:
                        # Swap direction if needed
                        return {
                            "type": relation_type,
                            "text": between_text,
                            "source_text": end_entity["text"],
                            "target_text": start_entity["text"],
                            "source_entity": end_entity,
                            "target_entity": start_entity,
                            "confidence": 0.8,
                            "source": "pattern"
                        }
        
        # If no pattern match, check with LLM if enabled
        if self.use_llm and self.llm_client:
            return self._determine_relation_with_llm(text, entity1, entity2)
        
        return None
    
    def _is_valid_relation_direction(self, source_entity: Dict[str, Any], target_entity: Dict[str, Any], relation_type: str) -> bool:
        """
        Check if the relation direction is valid based on entity types.
        
        Args:
            source_entity: Source entity
            target_entity: Target entity
            relation_type: Type of relation
            
        Returns:
            True if direction is valid, False otherwise
        """
        # Define expected source and target types for each relation
        relation_entity_types = {
            "TREATS": {
                "source": ["MEDICATION", "TREATMENT", "PROCEDURE"],
                "target": ["CONDITION", "DISEASE", "SYMPTOM"]
            },
            "PREVENTS": {
                "source": ["MEDICATION", "TREATMENT", "PROCEDURE", "VACCINE"],
                "target": ["CONDITION", "DISEASE", "SYMPTOM"]
            },
            "DIAGNOSES": {
                "source": ["TEST", "PROCEDURE"],
                "target": ["CONDITION", "DISEASE"]
            },
            "CAUSES": {
                "source": ["CONDITION", "MEDICATION", "PROCEDURE"],
                "target": ["SYMPTOM", "CONDITION", "EFFECT"]
            },
            "SUGGESTS": {
                "source": ["SYMPTOM", "TEST", "FINDING"],
                "target": ["CONDITION", "DISEASE"]
            }
        }
        
        # If relation type not defined, assume valid
        if relation_type not in relation_entity_types:
            return True
        
        # Check if entity types match expected types
        source_type = source_entity.get("label", "UNKNOWN")
        target_type = target_entity.get("label", "UNKNOWN")
        
        expected_sources = relation_entity_types[relation_type]["source"]
        expected_targets = relation_entity_types[relation_type]["target"]
        
        # If source and target match expected types, direction is valid
        if source_type in expected_sources and target_type in expected_targets:
            return True
        
        # If reverse direction matches expected types, direction is invalid
        if source_type in expected_targets and target_type in expected_sources:
            return False
        
        # Default to valid if can't determine
        return True

    def generate_relation_extraction(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate relation extraction patterns and extract relations between entities
        
        Args:
            text: Original text to analyze
            entities: List of entities to find relations between
            
        Returns:
            List of extracted relations
        """
        relations = []
        
        try:
            # Short circuit if we have fewer than 2 entities
            if len(entities) < 2:
                return []
            
            # Create a prompt for the LLM to extract relations
            entities_str = "\n".join([
                f"{i+1}. {entity.get('text', '')} (Type: {entity.get('entity_type', 'UNKNOWN')})"
                for i, entity in enumerate(entities[:10])  # Limit to first 10 entities
            ])
            
            prompt = f"""
            Extract medical relationships from the following text, focusing on the entities listed below:
            
            TEXT:
            {text[:1000]}  # Limit text length
            
            ENTITIES:
            {entities_str}
            
            For each pair of related entities, specify:
            - Source entity (from the list)
            - Target entity (from the list)
            - Relationship type (e.g., TREATS, CAUSES, INDICATES, DIAGNOSES, LOCATED_IN, etc.)
            - Confidence score (0.0 to 1.0)
            
            ONLY extract relationships that are explicitly or strongly implied in the text.
            Format each relationship as a JSON object.
            """
            
            # Use the LLM to extract relations
            response = self.llm_client.generate_response(prompt)
            
            try:
                # Try to parse the response as JSON
                import re
                import json
                
                # Find JSON objects in the response
                json_matches = re.findall(r'\{[^{}]*\}', response)
                
                for json_str in json_matches:
                    try:
                        rel_obj = json.loads(json_str)
                        
                        # Check if this is a valid relation object
                        if 'source' in rel_obj and 'target' in rel_obj and 'relationship_type' in rel_obj:
                            # Map source and target to actual entity texts
                            source_text = rel_obj['source']
                            target_text = rel_obj['target']
                            
                            # Find the corresponding entities
                            source_entity = None
                            target_entity = None
                            
                            for entity in entities:
                                if entity.get('text', '').lower() in source_text.lower() or source_text.lower() in entity.get('text', '').lower():
                                    source_entity = entity
                                if entity.get('text', '').lower() in target_text.lower() or target_text.lower() in entity.get('text', '').lower():
                                    target_entity = entity
                            
                            if source_entity and target_entity:
                                relations.append({
                                    'source': source_entity['text'],
                                    'source_id': source_entity.get('id', ''),
                                    'target': target_entity['text'],
                                    'target_id': target_entity.get('id', ''),
                                    'relation_type': rel_obj.get('relationship_type', 'RELATED_TO'),
                                    'confidence': rel_obj.get('confidence', 0.7),
                                    'method': 'llm-generated'
                                })
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
            
            return relations
        except Exception as e:
            print(f"Error in relation extraction generation: {e}")
            return []
    
    def _determine_relation_with_llm(self, text: str, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Use LLM to determine relationship between entities.
        
        Args:
            text: Context text
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Relation dictionary if found, None otherwise
        """
        try:
            # Create context window around entities (+-100 chars)
            start_pos = max(0, min(entity1["start"], entity2["start"]) - 100)
            end_pos = min(len(text), max(entity1["end"], entity2["end"]) + 100)
            context = text[start_pos:end_pos]
            
            # Format prompt for LLM
            prompt = f"""
            In this medical text: "{context}"
            
            What is the relationship between:
            1. {entity1['text']} ({entity1['label']})
            2. {entity2['text']} ({entity2['label']})
            
            Choose from these relation types: {', '.join(self.relation_types)}
            If no clear relation exists, respond with "NO_RELATION".
            
            Return your answer as a JSON object with the following fields:
            - relation_type: the type of relationship or "NO_RELATION"
            - direction: 1->2 or 2->1 (which entity is the source)
            - confidence: a value between 0 and 1
            """
            
            # Get LLM response
            response = self.llm_client.generate_relation_extraction(prompt)
            
            if isinstance(response, dict) and "relation_type" in response:
                relation_type = response["relation_type"]
                
                # Skip if no relation found
                if relation_type == "NO_RELATION":
                    return None
                
                # Determine source and target based on direction
                if response.get("direction") == "1->2":
                    source_entity = entity1
                    target_entity = entity2
                else:
                    source_entity = entity2
                    target_entity = entity1
                
                # Create relation
                return {
                    "type": relation_type,
                    "text": f"{source_entity['text']} {relation_type.lower()} {target_entity['text']}",
                    "source_text": source_entity["text"],
                    "target_text": target_entity["text"],
                    "source_entity": source_entity,
                    "target_entity": target_entity,
                    "confidence": response.get("confidence", 0.7),
                    "source": "llm"
                }
        
        except Exception as e:
            print(f"Error in LLM relation determination: {e}")
        
        return None
    
    def validate_knowledge_graph(self) -> bool:
        """
        Validate the constructed knowledge graph.
        
        Returns:
            True if graph is valid, False otherwise
        """
        # Empty graph is considered invalid
        if len(self.knowledge_graph.nodes()) == 0:
            return False
            
        # Check for disconnected components
        if not nx.is_weakly_connected(self.knowledge_graph) and len(self.knowledge_graph.nodes()) > 3:
            # Allow small graphs to be disconnected
            components = list(nx.weakly_connected_components(self.knowledge_graph))
            if len(components) > len(self.knowledge_graph.nodes()) / 3:
                return False  # Too fragmented
        
        # Check edge attributes
        for _, _, attrs in self.knowledge_graph.edges(data=True):
            if "relation" not in attrs:
                return False
        
        return True
    
    def map_entities_with_llm(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use LLM to map entities to standardized medical codes.
        
        Args:
            entities: List of entities to map
            
        Returns:
            Mapped entities with standardized codes
        """
        if not self.use_llm or not self.llm_client:
            return entities
            
        mapped_entities = []
        
        # Process in batches to avoid context length issues
        batch_size = 5
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i+batch_size]
            
            # Create prompt
            entity_descriptions = "\n".join([
                f"{j+1}. {e['text']} (Type: {e['label']})"
                for j, e in enumerate(batch)
            ])
            
            prompt = f"""
            Map these medical entities to standardized terminology and codes:
            
            {entity_descriptions}
            
            For each entity, provide:
            1. A preferred term
            2. Relevant codes (SNOMED CT, ICD-10, RxNorm, LOINC as applicable)
            3. Confidence score (0-1)
            
            Return as a JSON array with one object per entity.
            """
            
            try:
                # Get LLM response
                mappings = self.llm_client.generate_entity_mapping(prompt)
                
                if isinstance(mappings, list) and len(mappings) == len(batch):
                    # Update entities with mappings
                    for j, mapping in enumerate(mappings):
                        mapped_entity = batch[j].copy()
                        mapped_entity.update(mapping)
                        mapped_entities.append(mapped_entity)
                else:
                    # If mapping failed, keep original entities
                    mapped_entities.extend(batch)
                    
            except Exception as e:
                print(f"Error in LLM entity mapping: {e}")
                mapped_entities.extend(batch)
        
        return mapped_entities

    def extract_and_validate(self, text: str, entities: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Extract relations and validate the knowledge graph.
        
        Args:
            text: Text to analyze
            entities: Optional pre-extracted entities
            
        Returns:
            Dictionary with extracted relations and validation results
        """
        # Extract entities if not provided
        if not entities:
            entities = self.nlp.extract_entities(text)
            
        # Map entities to standardized codes if possible
        mapped_entities = self.map_entities_with_llm(entities) if self.use_llm else entities
        
        # Extract relations
        relations = self.extract_relations(text, mapped_entities)
        
        # Validate knowledge graph
        is_valid = self.validate_knowledge_graph()
        
        return {
            "entities": mapped_entities,
            "relations": relations,
            "is_valid": is_valid,
            "knowledge_graph": {
                "nodes": len(self.knowledge_graph.nodes()),
                "edges": len(self.knowledge_graph.edges())
            }
        }
        
    def get_knowledge_graph(self) -> nx.DiGraph:
        """
        Get the current knowledge graph.
        
        Returns:
            The knowledge graph
        """
        return self.knowledge_graph
        
    def reset_knowledge_graph(self) -> None:
        """Reset the knowledge graph"""
        self.knowledge_graph = nx.DiGraph()