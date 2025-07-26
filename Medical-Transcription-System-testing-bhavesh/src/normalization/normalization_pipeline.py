import os
import json
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import argparse
import traceback

# Import FastTerminologyResolver instead of ScispacyTerminologyResolver
try:
    from fast_terminology_resolver import FastTerminologyResolver
    RESOLVER_AVAILABLE = True
except ImportError:
    RESOLVER_AVAILABLE = False
    print("Error: Could not import FastTerminologyResolver")
    print("Please run: python src/normalization/build_terminology_index.py first")

def assign_confidence_scores(resolved_entities):
    """
    Assign confidence scores to resolved entities based on quality of mapping
    """
    for entity_text, entity_data in resolved_entities.items():
        # Default medium confidence
        confidence = 0.5
        
        # If we have a full code mapping, higher confidence
        if entity_data["codes"]:
            confidence = 0.8
            
            # If also has a good normalized form, even higher confidence
            if entity_data["normalized"] != entity_text:
                confidence = 0.9
        
        # If we couldn't map it at all, lower confidence
        else:
            confidence = 0.3
        
        entity_data["confidence"] = confidence
    
    return resolved_entities

def normalize_entities(data: Dict[str, Any], 
                      model_name: str = "biosent-clinical",
                      use_gpu: bool = False) -> Dict[str, Any]:
    """
    Normalize extracted medical entities to standard terminology codes
    
    Args:
        data: Dictionary containing entities to normalize
        model_name: Name of the embedding model to use
        use_gpu: Whether to use GPU for encoding
        
    Returns:
        Dictionary with normalized entities
    """
    if not data or not isinstance(data, dict):
        print("Error: Invalid input data for normalization")
        return None
        
    if not RESOLVER_AVAILABLE:
        print("Warning: Terminology resolver not available. Skipping normalization.")
        return data
    
    start_time = time.time()
    print(f"Normalizing entities using FastTerminologyResolver with model: {model_name}")
    
    # Extract entities from the data
    entities = []
    entity_types = {}
    
    # Check for speaker_turns structure which is the format in your data
    if "speaker_turns" in data:
        print(f"Found {len(data['speaker_turns'])} speaker turns")
        for i, turn in enumerate(data["speaker_turns"]):
            if "entities" in turn:
                print(f"Found {len(turn['entities'])} entities in speaker turn {i}")
                for entity in turn["entities"]:
                    if "text" in entity and entity["text"]:
                        entities.append(entity["text"])
                        # Use 'type' field if available, otherwise try 'label'
                        if "type" in entity and entity["type"]:
                            entity_types[entity["text"]] = entity["type"]
                        elif "label" in entity:
                            # Convert label like 'SYMPTOM' to lowercase 'symptom'
                            label = entity["label"].lower()
                            entity_types[entity["text"]] = label
    
    # Also check for direct entities field (as a fallback)
    elif "entities" in data:
        print(f"Found {len(data['entities'])} entities in top level")
        for entity in data["entities"]:
            if "text" in entity and entity["text"]:
                entities.append(entity["text"])
                if "type" in entity and entity["type"]:
                    entity_types[entity["text"]] = entity["type"]
                elif "label" in entity:
                    label = entity["label"].lower()
                    entity_types[entity["text"]] = label
    
    # Check for medications at top level or in speaker turns
    if "medications" in data:
        for med in data["medications"]:
            if "name" in med and med["name"]:
                entities.append(med["name"])
                entity_types[med["name"]] = "medication"
    else:
        # Check for medications in each speaker turn
        if "speaker_turns" in data:
            for turn in data["speaker_turns"]:
                if "medications" in turn:
                    for med in turn["medications"]:
                        if "name" in med and med["name"]:
                            entities.append(med["name"])
                            entity_types[med["name"]] = "medication"
    
    # Deduplicate entities
    entities = list(set(entities))
    
    if not entities:
        print("No entities found for normalization")
        data["normalized_entities"] = {}
        return data
    
    print(f"Found {len(entities)} unique entities to normalize")
    print(f"Entities: {entities}")
    print(f"Entity types: {entity_types}")
    
    # Initialize the resolver
    try:
        resolver = FastTerminologyResolver(model_name=model_name, use_gpu=use_gpu)
        resolved_entities = resolver.resolve_entities(entities, entity_types)
        resolver.close()
    except Exception as e:
        print(f"Error using FastTerminologyResolver: {e}")
        traceback.print_exc()
        data["normalized_entities"] = {}
        return data
    
    # Assign confidence scores
    resolved_entities = assign_confidence_scores(resolved_entities)
    
    # Add the normalized entities to the data
    data["normalized_entities"] = resolved_entities
    
    print(f"Entity normalization completed in {time.time() - start_time:.2f} seconds")
    return data

def process_file(input_file: str, 
                output_file: Optional[str] = None,
                model_name: str = "biosent-clinical",
                use_gpu: bool = False) -> Optional[Dict[str, Any]]:
    """
    Process a single input file
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (optional)
        model_name: Name of the embedding model to use
        use_gpu: Whether to use GPU for encoding
        
    Returns:
        Processed data dictionary
    """
    print(f"Processing file: {input_file}")
    
    # Check input file existence
    if not os.path.isfile(input_file):
        print(f"Error: Input file does not exist: {input_file}")
        return None
        
    # Read input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return None
    
    # Process data
    processed_data = normalize_entities(data, model_name, use_gpu)
    
    # Write output if needed
    if output_file and processed_data:
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            print(f"Wrote normalized data to: {output_file}")
        except Exception as e:
            print(f"Error writing output file: {e}")
    
    return processed_data

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Medical entity normalization pipeline")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", help="Output file or directory (default: input_normalized.json)")
    parser.add_argument("--model", default="biosent-clinical", help="Embedding model to use")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for embeddings")
    args = parser.parse_args()
    
    # Check if input is a directory or file
    if os.path.isdir(args.input):
        # Process directory
        input_dir = args.input
        output_dir = args.output or os.path.join(os.path.dirname(input_dir), 
                                                os.path.basename(input_dir) + "_normalized")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing directory: {input_dir} -> {output_dir}")
        
        # Find all JSON files
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        
        for filename in json_files:
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            process_file(input_file, output_file, args.model, args.gpu)
    else:
        # Process single file
        input_file = args.input
        
        if not args.output:
            # Generate default output filename
            name, ext = os.path.splitext(input_file)
            output_file = f"{name}_normalized{ext}"
        else:
            output_file = args.output
            
        process_file(input_file, output_file, args.model, args.gpu)
        
if __name__ == "__main__":
    main()