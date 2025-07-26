import os
import re
import json
import time
import pickle
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Handle dependencies gracefully
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Install with: pip install faiss-cpu (or faiss-gpu for CUDA support)")

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
    print("Using sentence-transformers for embeddings")
except ImportError:
    SBERT_AVAILABLE = False
    print("SentenceTransformer not available. Install with: pip install sentence-transformers")

class FastTerminologyResolver:
    """
    Resolves medical entities to standard terminology codes using vector embeddings
    and fast nearest neighbor search.
    """
    
    # Recommended biomedical models
    BIOMEDICAL_MODELS = {
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "biobert": "dmis-lab/biobert-v1.1",
        "scibert": "allenai/scibert_scivocab_uncased",
        "biosent-clinical": "pritamdeka/S-BioBert-snli-multinli-stsb",
        "biosent": "pritamdeka/BioBERT-mnli-snli-scinli",
        "biomed-roberta": "allenai/biomed_roberta_base",
        "default": "all-MiniLM-L6-v2"
    }
    
    def __init__(self, 
                 index_dir: Optional[str] = None,
                 model_name: str = "biosent-clinical",
                 force_rebuild: bool = False,
                 use_gpu: bool = False):
        """
        Initialize the fast terminology resolver
        
        Args:
            index_dir: Directory to store/load precomputed indexes
            model_name: Name of the model to use 
            force_rebuild: Whether to force rebuilding the index even if it exists
            use_gpu: Whether to use GPU for encoding (if available)
        """
        self.start_time = time.time()
        print(f"Initializing FastTerminologyResolver...")
        
        # Check requirements
        if not FAISS_AVAILABLE or not SBERT_AVAILABLE:
            print("WARNING: Missing dependencies. Terminology resolution will be limited.")
            return
        
        # Set up directories
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        if index_dir:
            self.index_dir = index_dir
        else:
            self.index_dir = os.path.join(self.script_dir, "data", "embeddings")
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Set up cache
        self.cache = {}
        
        # Initialize encoder
        if model_name in self.BIOMEDICAL_MODELS:
            self.model_name = self.BIOMEDICAL_MODELS[model_name]
            print(f"Using model shorthand '{model_name}' -> {self.model_name}")
        else:
            self.model_name = model_name
            
        # Try loading the model
        try:
            print(f"Loading model: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
            print(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Falling back to default model")
            self.model_name = self.BIOMEDICAL_MODELS["default"]
            try:
                self.encoder = SentenceTransformer(self.model_name)
            except Exception as e2:
                print(f"Could not load default model either: {e2}")
                print("Terminology resolution will be limited.")
                return
        
        # Get embedding dimension
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"Model loaded with embedding dimension: {self.embedding_dim}")
        
        # Use GPU if requested and available
        if use_gpu:
            try:
                self.encoder = self.encoder.to('cuda')
                print("Using GPU for encoding")
            except Exception as e:
                print(f"Could not use GPU: {e}")
                print("Falling back to CPU")
        
        # Load or build indexes
        self.indexes = {}
        self.terminology_data = {}
        terminology_types = ['conditions', 'medications', 'symptoms', 'procedures', 'labs']
        
        for term_type in terminology_types:
            self._load_or_build_index(term_type, force_rebuild)
            
        # Prepare mappings of common entity types
        self.entity_type_map = {
            'condition': 'conditions',
            'disease': 'conditions', 
            'diagnosis': 'conditions',
            'medication': 'medications',
            'drug': 'medications',
            'treatment': 'medications',
            'symptom': 'symptoms',
            'problem': 'symptoms',
            'sign': 'symptoms',
            'procedure': 'procedures',
            'operation': 'procedures',
            'surgery': 'procedures', 
            'lab': 'labs',
            'laboratory': 'labs',
            'test': 'labs',
            'test_result': 'labs'
        }
        
        print(f"FastTerminologyResolver initialized in {time.time() - self.start_time:.2f} seconds")
    
    def _load_or_build_index(self, terminology_type: str, force_rebuild: bool = False):
        """
        Load or build the FAISS index and terminology data for a given terminology type
        
        Args:
            terminology_type: Type of terminology (conditions, medications, etc.)
            force_rebuild: Whether to force rebuilding the index
        """
        # Include model name in index file path
        model_suffix = self.model_name.replace("/", "_").replace("-", "_")
        index_file = os.path.join(self.index_dir, f"{terminology_type}_{model_suffix}_index.bin")
        data_file = os.path.join(self.index_dir, f"{terminology_type}_data.pkl")
        
        # Check if index and data exist
        if not force_rebuild and os.path.exists(index_file) and os.path.exists(data_file):
            # Load existing index and data
            try:
                self.indexes[terminology_type] = faiss.read_index(index_file)
                with open(data_file, 'rb') as f:
                    self.terminology_data[terminology_type] = pickle.load(f)
                print(f"Loaded existing {terminology_type} index with {self.indexes[terminology_type].ntotal} entries")
                return
            except Exception as e:
                print(f"Error loading existing index: {e}")
                print(f"Rebuilding {terminology_type} index...")
        
        # Build new index
        print(f"Building {terminology_type} index...")
        terms, codes = self._load_terminology_data(terminology_type)
        
        if not terms:
            print(f"No data available for {terminology_type}")
            self.indexes[terminology_type] = None
            self.terminology_data[terminology_type] = {"terms": [], "codes": [], "model": self.model_name}
            return
        
        # Compute embeddings
        embeddings_matrix = self._encode_terms(terms)
        
        if embeddings_matrix is None or embeddings_matrix.shape[0] == 0:
            print(f"Failed to create embeddings for {terminology_type}")
            self.indexes[terminology_type] = None
            self.terminology_data[terminology_type] = {"terms": [], "codes": [], "model": self.model_name}
            return
        
        # Create FAISS index
        index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product = cosine on normalized vectors
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        # Add vectors to index
        index.add(embeddings_matrix)
        
        # Save index and data
        try:
            faiss.write_index(index, index_file)
            
            data = {
                "terms": terms,
                "codes": codes,
                "model": self.model_name
            }
            
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
                
            print(f"Saved {terminology_type} index to {index_file}")
        except Exception as e:
            print(f"Error saving index: {e}")
        
        # Store in memory
        self.indexes[terminology_type] = index
        self.terminology_data[terminology_type] = data
        
        print(f"Built and saved {terminology_type} index with {len(terms)} entries")
    
    def _encode_terms(self, terms: List[str]) -> np.ndarray:
        """
        Encode terms using the embedding method
        
        Args:
            terms: List of terms to encode
            
        Returns:
            NumPy array of embeddings
        """
        if not terms:
            return np.array([])
            
        # Use sentence-transformers for encoding
        batch_size = 64  # Smaller batch size to avoid memory issues
        all_embeddings = []
        
        for i in range(0, len(terms), batch_size):
            batch = terms[i:i + batch_size]
            embeddings = self.encoder.encode(batch, convert_to_numpy=True, show_progress_bar=i == 0)
            all_embeddings.append(embeddings)
            
            if i % 1000 == 0 and i > 0:
                print(f"Encoded {i} terms...")
        
        return np.vstack(all_embeddings)
    
    def _encode_single_term(self, term: str) -> np.ndarray:
        """
        Encode a single term
        
        Args:
            term: Term to encode
            
        Returns:
            NumPy array of embedding
        """
        return self.encoder.encode([term], convert_to_numpy=True)
    
    def _load_terminology_data(self, terminology_type: str) -> Tuple[List[str], List[Dict]]:
        """
        Load terminology data for a specific type
        
        Args:
            terminology_type: Type of terminology to load
            
        Returns:
            Tuple of (terms, codes)
        """
        # Paths to standard terminology files
        terminology_file = os.path.join(
            self.script_dir, "dictionaries", f"{terminology_type}.csv"
        )
        
        # Sample data for common medical terms if file doesn't exist
        samples = self._get_sample_data(terminology_type)
        
        # Load terminology data from file if exists
        if os.path.exists(terminology_file):
            try:
                # Load from CSV
                import csv
                terms = []
                codes = []
                
                with open(terminology_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        term = row.get('term', '')
                        if not term:
                            continue
                            
                        code_data = {}
                        # Extract codes based on terminology type
                        if terminology_type == 'medications' and 'rxnorm_code' in row:
                            code_data['rxnorm'] = row['rxnorm_code']
                        elif terminology_type == 'conditions' and 'icd10_code' in row:
                            code_data['icd10'] = row['icd10_code']
                        elif terminology_type == 'conditions' and 'snomed_code' in row:
                            code_data['snomed'] = row['snomed_code']
                        elif terminology_type == 'labs' and 'loinc_code' in row:
                            code_data['loinc'] = row['loinc_code']
                        
                        terms.append(term)
                        codes.append(code_data)
                
                # Add sample data as well for robustness
                terms.extend([s['term'] for s in samples])
                codes.extend([s['codes'] for s in samples])
                
                return terms, codes
            except Exception as e:
                print(f"Error loading terminology data from {terminology_file}: {e}")
        
        # Return sample data if no file exists
        return [s['term'] for s in samples], [s['codes'] for s in samples]
    
    def _get_sample_data(self, terminology_type: str) -> List[Dict]:
        """Get sample terminology data for a specific type"""
        sample_data = []
        
        if terminology_type == 'medications':
            sample_data = [
                {"term": "metformin", "codes": {"rxnorm": "6809", "snomed": "109081006"}},
                {"term": "metformin hydrochloride", "codes": {"rxnorm": "6809", "snomed": "109081006"}},
                {"term": "insulin", "codes": {"rxnorm": "5856", "snomed": "325072002"}},
                {"term": "lisinopril", "codes": {"rxnorm": "29046", "snomed": "108966004"}},
                {"term": "atorvastatin", "codes": {"rxnorm": "83367", "snomed": "373567001"}},
                {"term": "aspirin", "codes": {"rxnorm": "1191", "snomed": "372709000"}},
                {"term": "ibuprofen", "codes": {"rxnorm": "5640", "snomed": "387207008"}}
            ]
        elif terminology_type == 'conditions':
            sample_data = [
                {"term": "diabetes", "codes": {"snomed": "73211009", "icd10": "E11.9"}},
                {"term": "diabetes mellitus", "codes": {"snomed": "73211009", "icd10": "E11.9"}},
                {"term": "type 2 diabetes", "codes": {"snomed": "44054006", "icd10": "E11.9"}},
                {"term": "hypertension", "codes": {"snomed": "38341003", "icd10": "I10"}},
                {"term": "migraine", "codes": {"snomed": "37796009", "icd10": "G43.909"}},
                {"term": "headache", "codes": {"snomed": "25064002", "icd10": "R51"}}
            ]
        elif terminology_type == 'symptoms':
            sample_data = [
                {"term": "pain", "codes": {"snomed": "22253000", "icd10": "R52"}},
                {"term": "tingling", "codes": {"snomed": "62507009", "icd10": "R20.2"}},
                {"term": "paresthesia", "codes": {"snomed": "62507009", "icd10": "R20.2"}},
                {"term": "numbness", "codes": {"snomed": "44077006", "icd10": "R20.0"}},
                {"term": "dizziness", "codes": {"snomed": "404640003", "icd10": "R42"}},
                {"term": "lightheadedness", "codes": {"snomed": "386705008", "icd10": "R42"}},
                {"term": "fatigue", "codes": {"snomed": "84229001", "icd10": "R53.83"}},
                {"term": "sweating", "codes": {"snomed": "415690000", "icd10": "R61"}},
                {"term": "nausea", "codes": {"snomed": "422587007", "icd10": "R11.0"}},
                {"term": "photophobia", "codes": {"snomed": "13791008", "icd10": "H53.14"}},
                {"term": "sensitivity to light", "codes": {"snomed": "13791008", "icd10": "H53.14"}}
            ]
        elif terminology_type == 'labs':
            sample_data = [
                {"term": "hemoglobin a1c", "codes": {"loinc": "4548-4"}},
                {"term": "a1c", "codes": {"loinc": "4548-4"}},
                {"term": "blood glucose", "codes": {"loinc": "6749-6"}},
                {"term": "blood sugar", "codes": {"loinc": "6749-6"}},
                {"term": "complete blood count", "codes": {"loinc": "58410-2"}},
                {"term": "cbc", "codes": {"loinc": "58410-2"}}
            ]
        
        return sample_data
    
    def _normalize_term(self, term: str) -> str:
        """Normalize a term by removing extra spaces and making lowercase"""
        return re.sub(r'\s+', ' ', term.lower().strip())
    
    def _get_entity_category(self, entity_type: str) -> str:
        """Map entity type to one of our terminology categories"""
        if not entity_type:
            return None
            
        entity_type = entity_type.lower()
        return self.entity_type_map.get(entity_type)
    
    def resolve_term(self, 
                    term: str, 
                    entity_type: Optional[str] = None, 
                    top_k: int = 3) -> Dict[str, Any]:
        """
        Resolve a single term using vector similarity search
        
        Args:
            term: Term to resolve
            entity_type: Type of entity (medication, condition, etc.)
            top_k: Number of top matches to consider
            
        Returns:
            Dictionary with normalized form and codes
        """
        # Initial result structure
        result = {
            "original": term,
            "normalized": term,
            "codes": {},
            "type": entity_type
        }
        
        # Check cache first
        cache_key = f"{term}:{entity_type or 'unknown'}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Skip empty terms
        if not term or len(term.strip()) == 0:
            return result
        
        # Normalize term
        normalized_term = self._normalize_term(term)
        result["normalized"] = normalized_term
        
        # Determine which index to search based on entity type
        index_type = self._get_entity_category(entity_type)
        
        # If we can't determine the index type, try each index in priority order
        if not index_type:
            # Try different indexes in priority order
            index_types = ['medications', 'conditions', 'symptoms', 'labs']
            
            for idx_type in index_types:
                codes = self._search_index(normalized_term, idx_type, top_k)
                if codes:
                    result["codes"].update(codes)
        else:
            # Search the specific index type
            codes = self._search_index(normalized_term, index_type, top_k)
            if codes:
                result["codes"].update(codes)
        
        # Cache the result
        self.cache[cache_key] = result
        return result
    
    def _search_index(self, term: str, index_type: str, top_k: int = 3) -> Dict[str, str]:
        """
        Search a specific index for a term
        
        Args:
            term: Term to search for
            index_type: Type of index to search
            top_k: Number of top matches to consider
            
        Returns:
            Dictionary of terminology codes
        """
        if index_type not in self.indexes or self.indexes[index_type] is None:
            return {}
            
        # Get the embedding for the query term
        query_embedding = self._encode_single_term(term)
        
        # Normalize the vector
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        k = min(top_k, self.indexes[index_type].ntotal)  # Don't request more than we have
        if k == 0:
            return {}
            
        distances, indices = self.indexes[index_type].search(query_embedding, k)
        
        # Process results - check if we have a very good match
        if distances[0][0] > 0.9:
            idx = indices[0][0]
            if idx < len(self.terminology_data[index_type]["codes"]):
                return self.terminology_data[index_type]["codes"][idx]
        
        # Otherwise, check all top matches for an exact or very close match
        for i in range(min(k, len(indices[0]))):
            idx = indices[0][i]
            if idx < len(self.terminology_data[index_type]["terms"]):
                term_text = self.terminology_data[index_type]["terms"][idx]
                if term.lower() == term_text.lower() or distances[0][i] > 0.85:
                    return self.terminology_data[index_type]["codes"][idx]
        
        # If no good match, return the top match if it's reasonably close
        if distances[0][0] > 0.7:
            idx = indices[0][0]
            if idx < len(self.terminology_data[index_type]["codes"]):
                return self.terminology_data[index_type]["codes"][idx]
                
        return {}
    
    def resolve_entities(self, 
                        entities: List[str], 
                        entity_types: Optional[Dict[str, str]] = None) -> Dict[str, Dict]:
        """
        Resolve a list of entities to standard terminology codes
        
        Args:
            entities: List of entity strings to resolve
            entity_types: Dict mapping entity text to entity type
            
        Returns:
            Dict mapping entity text to normalized form and codes
        """
        entity_types = entity_types or {}
        results = {}
        
        for entity in entities:
            entity_type = entity_types.get(entity, None)
            results[entity] = self.resolve_term(entity, entity_type)
        
        return results
    
    def close(self):
        """Clean up resources"""
        print(f"FastTerminologyResolver closed. Cache size: {len(self.cache)} entries.")