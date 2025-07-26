import os
import argparse
from fast_terminology_resolver import FastTerminologyResolver

def list_available_models():
    """List the available biomedical models"""
    print("\nAvailable biomedical models (use shorthand name with --model):")
    print("-" * 80)
    print(f"{'Shorthand':<15} {'Full Model Name':<50} {'Size':<10}")
    print("-" * 80)
    
    models = [
        ("pubmedbert", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", "~400MB"),
        ("biobert", "dmis-lab/biobert-v1.1", "~400MB"),
        ("scibert", "allenai/scibert_scivocab_uncased", "~400MB"),
        ("biosent-clinical", "pritamdeka/S-BioBert-snli-multinli-stsb", "~400MB"),
        ("biosent", "pritamdeka/BioBERT-mnli-snli-scinli", "~400MB"),
        ("biomed-roberta", "allenai/biomed_roberta_base", "~400MB"),
        ("default", "all-MiniLM-L6-v2", "~80MB")
    ]
    
    for shorthand, full_name, size in models:
        print(f"{shorthand:<15} {full_name:<50} {size:<10}")
    print("-" * 80)
    print("Recommended: biosent-clinical (optimized for similarity in clinical text)")
    print()

def main():
    parser = argparse.ArgumentParser(description="Build terminology embedding indexes")
    parser.add_argument("--force", action="store_true", help="Force rebuild of all indexes")
    parser.add_argument("--model", default="biosent-clinical", 
                        help="Model to use (shorthand or full name)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for encoding")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    print(f"Building terminology indexes using model: {args.model}")
    print(f"Force rebuild: {args.force}")
    print(f"Using GPU: {args.gpu}")
    
    # Initialize the resolver which will build the indexes
    resolver = FastTerminologyResolver(
        model_name=args.model,
        force_rebuild=args.force,
        use_gpu=args.gpu
    )
    
    print("Indexes built successfully!")
    resolver.close()

if __name__ == "__main__":
    main()