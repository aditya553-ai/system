#!/usr/bin/env python3
import os
import sys
import argparse
from prescription_bridge import MedicalTranscriptionToPrescriptionBridge

def main():
    """Process transcripts and get prescription suggestions"""
    parser = argparse.ArgumentParser(
        description="Process medical transcripts and get prescription suggestions"
    )
    
    # Required arguments
    parser.add_argument(
        "transcript_path", 
        help="Path to transcript file (JSON or text)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--api-url", 
        default="http://localhost:8000", 
        help="URL for the auto-prescription API"
    )
    parser.add_argument(
        "--output-dir", 
        help="Directory to save output files"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate transcript path
    if not os.path.exists(args.transcript_path):
        print(f"Error: Transcript file not found: {args.transcript_path}")
        return 1
    
    # Create bridge
    bridge = MedicalTranscriptionToPrescriptionBridge(args.api_url)
    
    # Run pipeline
    result = bridge.run_full_pipeline(args.transcript_path, args.output_dir)
    
    if result:
        print("\nSuccessfully generated prescription suggestions!")
        
        # Display summary
        print("\n=== Prescription Summary ===")
        suggestions = result["prescription_suggestions"]
        
        print(f"Diagnosis: {suggestions.get('diagnosis', 'Not specified')}")
        print("\nRecommended Medications:")
        for medicine in suggestions.get("medicines", []):
            print(f"- {medicine['medicineName']} {medicine['dosage']}")
            print(f"  Instructions: {medicine['instructions']}")
            print(f"  Duration: {medicine['duration']} days")
        
        print(f"\nDoctor's Advice: {suggestions.get('doctorAdvice', 'None')[:200]}...")
        print(f"Follow-up Date: {suggestions.get('followUpDate', 'Not specified')}")
        
        print(f"\nFull results saved to: {result['suggestions_path']}")
        return 0
    else:
        print("Failed to generate prescription suggestions")
        return 1

if __name__ == "__main__":
    sys.exit(main())