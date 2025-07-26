import spacy
import re
from typing import Dict, List, Any, Optional, Tuple
from spacy.tokens import Doc, Span
import networkx as nx

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    LLM_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available for advanced relation extraction")
    LLM_AVAILABLE = False

class SemanticRelationExtractor:
    """
    Extract semantic relationships between medical entities in text
    using rule-based and optional LLM-enhanced approaches.
    """
    
    def __init__(self, nlp=None, use_llm=False):
        """
        Initialize the relation extractor
        
        Args:
            nlp: Optional spaCy pipeline
            use_llm: Whether to use LLM for enhanced relation extraction
        """
        # Initialize spaCy if not provided
        self.nlp = nlp or spacy.load("en_core_web_sm")
        
        # Define relation types and their markers
        self.relation_types = {
            "TREATS": ["treat", "treats", "treating", "for", "prescribed for", "helps with", "to manage", "helps manage"],
            "PREVENTS": ["prevent", "prevents", "preventing", "to avoid", "stops", "inhibits"],
            "DIAGNOSES": ["diagnose", "diagnoses", "diagnostic for", "confirms", "to detect", "detects", "identified"],
            "CAUSES": ["causes", "cause", "causing", "leads to", "results in", "responsible for", "triggers"],
            "SUGGESTS": ["suggests", "indicating", "indicates", "sign of", "symptom of", "consistent with"],
            "LOCATED_IN": ["in", "on", "at", "within", "located in", "located on", "located at"],
            "PART_OF": ["part of", "component of", "segment of", "portion of", "section of"],
            "EXPERIENCED_BY": ["experienced by", "suffered by", "reported by", "patient has", "patient reports", "I have"],
            "HAS_FREQUENCY": ["daily", "weekly", "monthly", "every", "times a day", "times per", "twice", "once"],
            "HAS_DOSAGE": ["mg", "milligram", "gram", "mcg", "microgram", "dose", "tablet", "capsule", "injection"],
            "HAS_DURATION": ["for", "during", "over", "throughout", "lasting", "persisting for", "since", "continues for"],
            "WORSENS": ["worsens", "aggravates", "exacerbates", "makes worse", "increases", "elevates"],
            "IMPROVES": ["improves", "alleviates", "helps", "reduces", "decreases", "diminishes", "relieves"]
        }
        
        # Context-specific relation patterns for speaker awareness
        self.speaker_patterns = {
            "PATIENT_HAS": [
                r"(?:I|patient|she|he) (?:have|has|experiencing|report[s]?|feel[s]?|felt|experiencing) (.*)",
                r"(?:I|patient|she|he)'m having (.*)",
                r"(?:I|patient|she|he) get (.*)",
                r"(?:My|patient's|her|his) (.*) (?:is|are|feels|feel) (.*)"
            ],
            "DOCTOR_SAYS": [
                r"(?:You|Dr\.|doctor|physician) (?:mentioned|said|noted|recorded|documented) (.*)",
                r"(?:You|Dr\.|doctor|physician) (?:think|believe|suspect) (.*)",
                r"(?:Your|Dr\.'s|doctor's) (?:diagnosis|assessment|evaluation|opinion) (?:is|was) (.*)"
            ]
        }
        
        # Entity type compatibility matrix for relations
        self.compatibility = {
            "TREATS": {
                "source": ["MEDICATION", "PROCEDURE", "TREATMENT"],
                "target": ["CONDITION", "SYMPTOM", "DISEASE", "DIAGNOSIS"]
            },
            "PREVENTS": {
                "source": ["MEDICATION", "PROCEDURE", "TREATMENT", "BEHAVIOR"],
                "target": ["CONDITION", "SYMPTOM", "DISEASE", "DIAGNOSIS"]
            },
            "CAUSES": {
                "source": ["CONDITION", "MEDICATION", "DISEASE", "BEHAVIOR"],
                "target": ["SYMPTOM", "CONDITION", "FINDING"]
            },
            "LOCATED_IN": {
                "source": ["SYMPTOM", "CONDITION", "FINDING", "DISEASE"],
                "target": ["ANATOMY", "BODY_PART", "LOCATION"]
            },
            "HAS_FREQUENCY": {
                "source": ["MEDICATION", "SYMPTOM", "PROCEDURE", "TREATMENT", "BEHAVIOR"],
                "target": ["TIME", "FREQUENCY", "DATE"]
            },
            "HAS_DOSAGE": {
                "source": ["MEDICATION", "TREATMENT"],
                "target": ["DOSAGE", "QUANTITY", "CARDINAL", "NUMBER"]
            },
            "HAS_DURATION": {
                "source": ["SYMPTOM", "CONDITION", "MEDICATION", "TREATMENT", "PROCEDURE"],
                "target": ["TIME", "DATE", "DURATION"]
            }
        }
        
        # Initialize LLM if available and requested
        self.use_llm = use_llm and LLM_AVAILABLE
        self.llm_model = None
        self.llm_tokenizer = None
        
        if self.use_llm:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM for enhanced relation extraction"""
        try:
            # Using a small relation classification model
            model_name = "bvanaken/clinical-assertion-negation-bert"
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.llm_model = self.llm_model.to("cuda")
                
            print(f"Initialized LLM for relation extraction: {model_name}")
        except Exception as e:
            print(f"Failed to initialize LLM for relation extraction: {e}")
            self.use_llm = False
    
    def extract_relations(self, doc, entities):
        """
        Extract semantic relations between entities in text, designed for
        processing entities within a SINGLE speaker turn
        
        Args:
            doc: spaCy Doc object for this turn
            entities: List of extracted entities for this turn
            
        Returns:
            Dictionary with extracted relations for this turn only
        """
        if not entities or len(entities) < 2:
            return {"relations": [], "count": 0}
        
        relations = []
        text = doc.text
        
        # For entity pairs within this turn, extract potential relations
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i == j:
                    continue
                
                # Extract entity information
                e1_text = entity1.get("text", "")
                e1_label = entity1.get("label", "")
                e1_start = entity1.get("start", 0)
                e1_end = entity1.get("end", 0)
                
                e2_text = entity2.get("text", "")
                e2_label = entity2.get("label", "")
                e2_start = entity2.get("start", 0)
                e2_end = entity2.get("end", 0)
                
                # Skip if either entity is empty
                if not e1_text or not e2_text:
                    continue
                    
                # Get relationship between entities
                relation_type, context_phrase = self._determine_relation(
                    text, 
                    e1_text, e1_label, e1_start, e1_end,
                    e2_text, e2_label, e2_start, e2_end,
                    doc
                )
                
                # If a relation was found, add it to the list
                if relation_type:
                    relation = {
                        "source": {"text": e1_text, "label": e1_label},
                        "target": {"text": e2_text, "label": e2_label},
                        "type": relation_type,
                        "confidence": 0.8,
                        "context": context_phrase
                    }
                    
                    # Check for duplicates before adding
                    if not self._is_duplicate_relation(relations, relation):
                        relations.append(relation)
        
        return {"relations": relations, "count": len(relations)}
    
    def _determine_relation(self, text, e1_text, e1_label, e1_start, e1_end, 
                           e2_text, e2_label, e2_start, e2_end, doc):
        """
        Determine the relationship between two entities
        
        Args:
            text: Full text
            e1_text, e1_label, e1_start, e1_end: First entity details
            e2_text, e2_label, e2_start, e2_end: Second entity details
            doc: spaCy Doc object
            
        Returns:
            Tuple of (relation_type, context_phrase) or (None, None) if no relation
        """
        # 1. Check for compatibility of entity types
        for relation_type, compatibility in self.compatibility.items():
            if (e1_label in compatibility["source"] and e2_label in compatibility["target"]):
                # Check if the entities are in the right order
                if e1_start < e2_start:
                    # Get the text between the two entities
                    between_text = text[e1_end:e2_start]
                    
                    # Check for relation markers
                    for marker in self.relation_types.get(relation_type, []):
                        if marker.lower() in between_text.lower():
                            # Extract context
                            context_start = max(0, e1_start - 20)
                            context_end = min(len(text), e2_end + 20)
                            context = text[context_start:context_end]
                            return relation_type, context
        
        # 2. Check for special relationships based on entity types
        
        # A. Anatomy-Symptom relationship
        if (e1_label == "SYMPTOM" and e2_label in ["ANATOMY", "BODY_PART"]):
            # Check if the symptom is located in the body part
            window_text = text[max(0, e1_start-30):min(len(text), e2_end+30)]
            if any(marker in window_text.lower() for marker in ["in", "on", "at", "of"]):
                return "LOCATED_IN", window_text
        
        # B. Medication-Dosage relationship
        if (e1_label in ["MEDICATION", "DRUG"] and e2_label in ["DOSAGE", "QUANTITY"]):
            # Check if they're close to each other
            if abs(e1_end - e2_start) < 15 or abs(e2_end - e1_start) < 15:
                context_start = min(e1_start, e2_start) - 5
                context_end = max(e1_end, e2_end) + 5
                context = text[max(0, context_start):min(len(text), context_end)]
                return "HAS_DOSAGE", context
        
        # C. Medication-Frequency relationship
        if (e1_label in ["MEDICATION", "DRUG"] and e2_label in ["FREQUENCY", "TIME"]):
            # Check if they're in the same sentence or close by
            if abs(e1_end - e2_start) < 50 or abs(e2_end - e1_start) < 50:
                # Look for frequency markers
                window_text = text[max(0, min(e1_start, e2_start)-20):min(len(text), max(e1_end, e2_end)+20)]
                for marker in self.relation_types.get("HAS_FREQUENCY", []):
                    if marker.lower() in window_text.lower():
                        return "HAS_FREQUENCY", window_text
        
        # D. Symptom-Time relationship
        if (e1_label == "SYMPTOM" and e2_label in ["TIME", "DATE", "DURATION"]):
            # Check if symptom occurs at specific time
            window_text = text[max(0, min(e1_start, e2_start)-20):min(len(text), max(e1_end, e2_end)+20)]
            return "OCCURS_AT", window_text
        
        # 3. Use dependency parsing to find implicit relationships
        # Find the sentence containing both entities
        containing_sent = None
        for sent in doc.sents:
            if (sent.start_char <= e1_start <= sent.end_char and 
                sent.start_char <= e2_start <= sent.end_char):
                containing_sent = sent
                break
        
        if containing_sent:
            # Find the dependency path between entities
            e1_token = None
            e2_token = None
            
            # Find tokens corresponding to entities
            for token in containing_sent:
                if e1_start <= token.idx < e1_end:
                    e1_token = token
                if e2_start <= token.idx < e2_end:
                    e2_token = token
            
            if e1_token and e2_token:
                # Check for direct dependency
                if e2_token.head == e1_token or e1_token.head == e2_token:
                    # Extract the dependency label
                    dep_label = e2_token.dep_ if e2_token.head == e1_token else e1_token.dep_
                    
                    # Map dependency to relation type
                    if dep_label in ["prep", "nmod"]:
                        return "LOCATED_IN", containing_sent.text
                    elif dep_label == "poss":
                        return "PART_OF", containing_sent.text
                    elif dep_label in ["nsubj", "dobj"]:
                        # Check the verb
                        verb = e2_token.head if e2_token.head != e1_token else e1_token.head
                        if verb.lemma_ in ["cause", "trigger", "lead"]:
                            return "CAUSES", containing_sent.text
                        elif verb.lemma_ in ["treat", "help", "alleviate", "reduce"]:
                            return "TREATS", containing_sent.text
        
        # No relation found
        return None, None
    
    def _extract_relations_with_llm(self, doc, entities):
        """Use LLM to extract relations between entities"""
        if not self.llm_model or not entities:
            return []
            
        relations = []
        
        try:
            # Prepare text and entities for LLM
            text = doc.text
            
            # For simplicity, consider only entity pairs with high potential for relations
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities):
                    if i == j:
                        continue
                    
                    # Extract entity information
                    e1_text = entity1.get("text", "")
                    e1_label = entity1.get("label", "")
                    e1_start = entity1.get("start", 0)
                    e1_end = entity1.get("end", 0)
                    
                    e2_text = entity2.get("text", "")
                    e2_label = entity2.get("label", "")
                    e2_start = entity2.get("start", 0)
                    e2_end = entity2.get("end", 0)
                    
                    # Skip if either entity is empty
                    if not e1_text or not e2_text:
                        continue
                    
                    # Get sentence containing both entities if possible
                    sent_text = text
                    for sent in doc.sents:
                        if (e1_start >= sent.start_char and e1_start < sent.end_char and
                            e2_start >= sent.start_char and e2_start < sent.end_char):
                            sent_text = sent.text
                            break
                    
                    # Use the LLM to determine the relation
                    inputs = self.llm_tokenizer(
                        f"Entity 1: {e1_text} ({e1_label}), Entity 2: {e2_text} ({e2_label}). Sentence: {sent_text}",
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    )
                    
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.llm_model(**inputs)
                        predictions = outputs.logits.softmax(dim=-1)
                        predicted_class = predictions.argmax().item()
                    
                    # Map class to relation type
                    relation_types = ["NONE", "TREATS", "CAUSES", "LOCATED_IN", "PART_OF", "HAS_FREQUENCY"]
                    relation_type = relation_types[predicted_class] if predicted_class < len(relation_types) else None
                    
                    if relation_type and relation_type != "NONE":
                        relation = {
                            "source": {"text": e1_text, "label": e1_label},
                            "target": {"text": e2_text, "label": e2_label},
                            "type": relation_type,
                            "confidence": predictions[0][predicted_class].item(),
                            "context": sent_text
                        }
                        relations.append(relation)
            
        except Exception as e:
            print(f"LLM relation extraction failed: {e}")
        
        return relations
    
    def _is_duplicate_relation(self, relations, new_relation):
        """
        Check if a relation already exists in the list
        
        Args:
            relations: List of existing relations
            new_relation: New relation to check
            
        Returns:
            Boolean indicating if the new relation is a duplicate
        """
        for rel in relations:
            if (rel["source"]["text"] == new_relation["source"]["text"] and
                rel["target"]["text"] == new_relation["target"]["text"] and
                rel["type"] == new_relation["type"]):
                return True
        return False
    
        # Update the extract_speaker_relationships method
    def extract_speaker_relationships(self, doc, entities, speaker_id=None):
        """Extract relationships specific to speaker context"""
        # This method helps determine if a symptom is experienced by the patient,
        # mentioned by the doctor, etc.
        
        if not entities:
            return []
        
        speaker_relations = []
        text = doc.text
        
        # Determine if text contains indicators of speaker type
        # Fix: Check if search returns a match object (not None)
        is_patient_speaking = bool(re.search(r'\b(I have|I am|I feel|my)\b', text, re.I))
        is_doctor_speaking = bool(re.search(r'\b(you have|you are|your)\b', text, re.I))
        
        # If speaker_id is provided, use it to enhance accuracy
        if speaker_id is not None:
            # In a typical medical conversation, speaker 0 might be doctor, 1 might be patient
            is_doctor_speaking = is_doctor_speaking or speaker_id == 0
            is_patient_speaking = is_patient_speaking or speaker_id == 1
        
        # Process each entity
        for entity in entities:
            entity_text = entity.get("text", "")
            entity_label = entity.get("label", "")
            entity_start = entity.get("start", 0)
            entity_end = entity.get("end", 0)
            
            if not entity_text:
                continue
            
            # Context window
            context_start = max(0, entity_start - 50)
            context_end = min(len(text), entity_end + 50)
            context = text[context_start:context_end]
            
            # For symptoms, determine who is experiencing them
            if entity_label in ["SYMPTOM", "CONDITION", "DISEASE", "DIAGNOSIS"]:
                if is_patient_speaking:
                    # Fix: Use bool() to check if search found a match
                    if bool(re.search(r'\b(I have|I feel|I am|I get|my)\b.*' + re.escape(entity_text), context, re.I)):
                        speaker_relations.append({
                            "source": {"text": "Patient", "label": "PERSON"},
                            "target": {"text": entity_text, "label": entity_label},
                            "type": "EXPERIENCES",
                            "confidence": 0.9,
                            "context": context
                        })
                elif is_doctor_speaking:
                    # Check if doctor is asking about a symptom
                    if bool(re.search(r'\b(do you have|are you feeling|have you noticed)\b.*' + re.escape(entity_text), context, re.I)):
                        speaker_relations.append({
                            "source": {"text": "Doctor", "label": "PERSON"},
                            "target": {"text": entity_text, "label": entity_label},
                            "type": "INQUIRES_ABOUT",
                            "confidence": 0.8,
                            "context": context
                        })
            
            # For medications, determine who is prescribing/taking them
            elif entity_label in ["MEDICATION", "DRUG", "TREATMENT"]:
                if is_doctor_speaking:
                    if bool(re.search(r'\b(I recommend|I suggest|you should take|we can try|let\'s start)\b.*' + re.escape(entity_text), context, re.I)):
                        speaker_relations.append({
                            "source": {"text": "Doctor", "label": "PERSON"},
                            "target": {"text": entity_text, "label": entity_label},
                            "type": "RECOMMENDS",
                            "confidence": 0.9,
                            "context": context
                        })
                elif is_patient_speaking:
                    if bool(re.search(r'\b(I take|I am on|I use|my medication|I started)\b.*' + re.escape(entity_text), context, re.I)):
                        speaker_relations.append({
                            "source": {"text": "Patient", "label": "PERSON"},
                            "target": {"text": entity_text, "label": entity_label},
                            "type": "TAKES",
                            "confidence": 0.9,
                            "context": context
                        })
        
        return speaker_relations
                
    def _find_temporal_references(self, text: str, entity_text: str) -> List[Tuple[str, str]]:
        """Find temporal references related to an entity in text"""
        temporal_refs = []
        
        # Check if entity is actually mentioned in the text
        if entity_text not in text:
            return temporal_refs
            
        # Check for duration patterns
        for pattern in self.temporal_patterns["duration"]:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Check if entity is mentioned nearby (within reasonable distance)
                entity_pos = text.find(entity_text)
                match_pos = match.start()
                if abs(entity_pos - match_pos) < 100:  # Within ~100 chars
                    temporal_refs.append(("duration", match.group(1)))
                    
        # Check for frequency patterns
        for pattern in self.temporal_patterns["frequency"]:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Check if entity is mentioned nearby
                entity_pos = text.find(entity_text)
                match_pos = match.start()
                if abs(entity_pos - match_pos) < 100:
                    temporal_refs.append(("frequency", match.group(0)))
                    
        # Check for specific time patterns
        for pattern in self.temporal_patterns["specific_time"]:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Check if entity is mentioned nearby
                entity_pos = text.find(entity_text)
                match_pos = match.start()
                if abs(entity_pos - match_pos) < 100:
                    temporal_refs.append(("specific_time", match.group(0)))
                    
        return temporal_refs
    
    def _check_entity_relationship(self, text: str, entity1: str, entity2: str) -> Tuple[Optional[str], Optional[str]]:
        """Check if there's a relationship between two entities in the text"""
        # If both entities are far apart, unlikely to be directly related
        e1_pos = text.find(entity1)
        e2_pos = text.find(entity2)
        
        if e1_pos == -1 or e2_pos == -1:
            return None, None
            
        # Get the text between the two entities
        if e1_pos < e2_pos:
            between_text = text[e1_pos + len(entity1):e2_pos]
        else:
            between_text = text[e2_pos + len(entity2):e1_pos]
            
        # Check for relationships in between text
        for rel_type, markers in self.relation_types.items():
            for marker in markers:
                if marker in between_text:
                    # Extract the specific phrase containing the relation
                    start = between_text.find(marker)
                    end = start + len(marker)
                    
                    # Get some context around the marker
                    phrase_start = max(0, start - 10)
                    phrase_end = min(len(between_text), end + 10)
                    phrase = between_text[phrase_start:phrase_end]
                    
                    return rel_type, phrase
                    
        return None, None