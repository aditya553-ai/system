from typing import Dict, List, Any, Optional
import json
import re
import copy
import threading
import time
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from fuzzywuzzy import process, fuzz # Added fuzz for similarity scoring
from model_manager import ModelManager
# Suppress a very verbose warning from HuggingFace tokenizers
# that can be spammy if the tokenizer doesn't have a specific fast version.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TranscriptionErrorCorrector:
    """
    Class for detecting and correcting errors in medical transcriptions
    using a hybrid approach with terminology resolvers and LLM-guided mapping.
    """
    
    def __init__(self, model_manager_instance: ModelManager, use_llm: bool = True):
        """
        Initialize the error corrector
        
        Args:
            use_llm: Whether to use LLM for corrections
        """
        self.use_llm = use_llm
        self.llm_model = None
        self.llm_tokenizer = None
        self.cache = {}  # Simple in-memory cache
        self.cache_lock = threading.Lock()
        
        self._load_medical_dictionaries()
        
        if self.use_llm:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM for error correction"""
        try:
            print("Initializing LLM for error correction...")
            
            # Prioritize medical-specific Seq2Seq models, then general Seq2Seq models
            model_candidates = [
                "GanjinZero/biobart-v2-base",    # Biomedical BART model (Seq2Seq)
                "t5-small",                      # General purpose Seq2Seq (fallback)
                "google/flan-t5-small"           # Another general purpose Seq2Seq (fallback)
            ]
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Attempting to load LLM on device: {device}")

            for model_name in model_candidates:
                try:
                    print(f"Attempting to load model: {model_name}")
                    self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    # For Seq2Seq models, AutoModelForSeq2SeqLM is appropriate.
                    # device_map="auto" is good for multi-gpu or large models.
                    # For single GPU/CPU, .to(device) is also common.
                    self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        device_map="auto" if device == "cuda" else None # device_map only for CUDA
                    )
                    if device == "cpu" and self.llm_model: # If on CPU, ensure model is explicitly moved
                        self.llm_model.to(device)

                    print(f"LLM initialized successfully: {model_name} on {self.llm_model.device if self.llm_model else 'N/A'}")
                    break
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
                    self.llm_model = None # Ensure it's reset on failure
                    self.llm_tokenizer = None
                    continue
            
            if self.llm_model is None:
                print("Could not load any LLM model. LLM-based correction will be disabled.")
                self.use_llm = False
                
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            self.use_llm = False
            self.llm_model = None
            self.llm_tokenizer = None

    def _load_medical_dictionaries(self):
        """Load medical terminology dictionaries for error correction"""
        self.medical_terms = {
            "medications": [
                "metformin", "insulin", "lisinopril", "atorvastatin", "simvastatin",
                "amlodipine", "metoprolol", "losartan", "hydrochlorothiazide", "omeprazole",
                "levothyroxine", "aspirin", "ibuprofen", "acetaminophen", "warfarin",
                "clopidogrel", "furosemide", "gabapentin", "prednisone", "fluticasone",
                "phenelzine" # Added for phonetic example
            ],
            "conditions": [
                "diabetes", "hypertension", "hyperlipidemia", "asthma", "COPD",
                "arthritis", "depression", "anxiety", "hypothyroidism", "coronary artery disease",
                "heart failure", "atrial fibrillation", "stroke", "chronic kidney disease",
                "osteoporosis", "GERD", "obstructive sleep apnea", "obesity", "cancer", "dementia"
            ],
            "symptoms": [
                "pain", "fatigue", "fever", "cough", "shortness of breath", 
                "nausea", "vomiting", "diarrhea", "constipation", "headache",
                "dizziness", "chest pain", "abdominal pain", "back pain", "joint pain",
                "weakness", "numbness", "tingling", "swelling", "rash"
            ],
            "medical_descriptors": [ # Added category for terms like 'erythematous'
                "erythematous", "tympanic", "myocardial", "coronary", "chronic", "acute",
                "palpable", "nonpalpable", "edematous"
            ]
        }
        
        self.common_errors = {
            # Medication misspellings and split words
            "metfomin": "metformin",
            "metphormin": "metformin",
            "metform in": "metformin", # Handles "metform in"
            "amlopidine": "amlodipine",
            "metropolol": "metoprolol",
            "lipator": "lipitor", 
            "simvastatine": "simvastatin",
            "furosamide": "furosemide",
            "warferin": "warfarin",
            "levo thyroxine": "levothyroxine",
            "hydrochlor thiazide": "hydrochlorothiazide",
            
            # Condition misspellings
            "diabetis": "diabetes",
            "diabeties": "diabetes",
            "hypertenshion": "hypertension",
            "hypertention": "hypertension",
            "arthritus": "arthritis",
            "hiperlipidemia": "hyperlipidemia",
            "atrial fibrilation": "atrial fibrillation",
            "atriel fibrilation": "atrial fibrillation",
            "asma": "asthma",
            "ashma": "asthma",
            
            # Common medical phrase errors
            "hart attack": "heart attack",
            "blood preshure": "blood pressure",
            "corinary": "coronary",
            "myocordial": "myocardial",
            "infraction": "infarction", # Common error, though 'infarction' better than 'infraction'
            "kidny": "kidney",
            "colestrol": "cholesterol",

            # Phonetic misspellings (can also be handled by LLM, but good to have common ones)
            "airy thameters": "erythematous", # Specific example from user
            "fenelzine": "phenelzine", # Example for LLM prompt
        }
        
        # Homophones - not actively used in correction logic yet, but kept for future
        self.homophones = {
            "throws": ["throes"], "role": ["roll"], "patients": ["patience"],
            "palate": ["palette", "pallet"], "flu": ["flew"], "vein": ["vain", "vane"],
            "ileum": ["ilium"], "murmur": ["murmor"], "stationary": ["stationery"],
            "sects": ["sex"], "bowel": ["bowl"], "serial": ["cereal"]
        }

    def correct_transcript(self, transcript_data: Dict) -> Dict:
        if not transcript_data or "speaker_turns" not in transcript_data:
            print("Warning: Transcript data is empty or malformed.")
            return transcript_data
            
        corrected_data = copy.deepcopy(transcript_data)
        corrections_made = 0
        llm_corrections_count = 0
        
        print("Processing transcript for corrections...")
        
        for i, turn in enumerate(corrected_data.get("speaker_turns", [])):
            if "text" not in turn or not turn["text"].strip():
                continue
                
            original_text = turn["text"]
            corrected_text = original_text
            
            # Stage 1: Common errors (dictionary-based, high precision)
            corrected_text = self.correct_common_errors(corrected_text)
            
            # Stage 2: Numerical expressions and dosage formatting
            corrected_text = self.correct_numbers(corrected_text)
            
            # Stage 3: Fuzzy matching for medical terms
            corrected_text = self.correct_medical_terms(corrected_text)
            
            # Stage 4: LLM-based correction for more complex errors
            if self.use_llm and self.llm_model and self.llm_tokenizer:
                speaker_role = turn.get("speaker_role", "unknown")
                # Pass the already partially corrected text to the LLM
                llm_candidate_text = self.correct_with_llm(
                    corrected_text,
                    speaker_role,
                    transcript_data.get("medical_context", "")
                )
                
                # Accept LLM correction if it's different and deemed valid
                if llm_candidate_text != corrected_text and \
                   self._is_valid_correction(corrected_text, llm_candidate_text):
                    corrected_text = llm_candidate_text
                    llm_corrections_count +=1
            
            if corrected_text != original_text:
                turn["original_text"] = original_text
                turn["text"] = corrected_text
                corrections_made += 1
        
        corrected_data["correction_stats"] = {
            "total_turns_processed": len(corrected_data.get("speaker_turns", [])),
            "total_turns_corrected": corrections_made,
            "llm_based_corrections": llm_corrections_count,
            "llm_enabled": self.use_llm and self.llm_model is not None,
            "correction_ratio": (corrections_made / len(corrected_data.get("speaker_turns", [])))
                                if corrected_data.get("speaker_turns") and len(corrected_data.get("speaker_turns", [])) > 0 else 0,
        }
        
        print(f"Correction complete. {corrections_made} turns corrected. LLM applied to {llm_corrections_count} turns.")
        return corrected_data
    
    def correct_with_llm(self, text: str, speaker_role: Optional[str] = None, medical_context: Optional[str] = None) -> str:
        if not self.use_llm or not self.llm_model or not self.llm_tokenizer:
            return text
        if not text.strip(): # Don't process empty strings
            return text
        
        cache_key = f"llm_{text}_{speaker_role or ''}_{medical_context or ''}"
        with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        try:
            prompt = f"""
            TASK: Correct medical transcription errors in the following text. Focus on accuracy and preserving original meaning.

            INSTRUCTIONS:
            - Correct misspellings of medical terms, medications, and anatomical names.
            - Fix severe phonetic transcriptions (e.g., "airy thameters" to "erythematous", "fenelzine" to "phenelzine").
            - Join split words that should be single terms (e.g., "metform in" to "metformin", "hydrochlor thiazide" to "hydrochlorothiazide").
            - Correct common grammatical errors ONLY if they obscure medical meaning; primarily focus on terminology.
            - Preserve medical measurements, dosages, and units accurately.
            - Do NOT add explanatory phrases like "The corrected text is:".
            - Do NOT change the core meaning or add new medical information.
            - Output ONLY the fully corrected text.

            ORIGINAL TEXT: {text}

            CORRECTED TEXT:
            """
            
            inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Move inputs to the model's device
            inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
                
            outputs = self.llm_model.generate(
                **inputs,
                max_length=max(len(text.split()) * 3, 150), # Dynamic max_length based on input
                min_length=len(text.split()) // 2, # Ensure not too short
                temperature=0.2,
                do_sample=False, # Greedy decoding for consistency
                num_beams=3, # Beam search can improve quality
                early_stopping=True,
                num_return_sequences=1
            )
            
            llm_output_text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Post-process to remove any residual prompt artifacts
            if "CORRECTED TEXT:" in llm_output_text: # Handles cases where model might repeat part of the prompt
                 llm_output_text = llm_output_text.split("CORRECTED TEXT:")[-1].strip()
            # Remove common instruction phrases if accidentally generated
            llm_output_text = re.sub(r'^(Corrected text:|Here is the corrected text:|The corrected version is:)\s*', '', llm_output_text, flags=re.IGNORECASE)


            # Basic check to prevent model from outputting empty string or placeholder
            if not llm_output_text.strip() or llm_output_text.lower() == "corrected text:":
                print(f"LLM produced invalid/empty output for: {text}")
                return text


            with self.cache_lock:
                self.cache[cache_key] = llm_output_text
            return llm_output_text
                
        except Exception as e:
            print(f"Error in LLM correction for text '{text[:50]}...': {e}")
            # import traceback
            # traceback.print_exc() # For debugging if needed
            return text # Fallback to text before LLM

    def correct_common_errors(self, text: str) -> str:
        if not text: return text
        
        # Sort errors by length of the error phrase (longest first)
        # to prevent shorter substrings from being replaced incorrectly.
        sorted_error_phrases = sorted(self.common_errors.keys(), key=len, reverse=True)
        
        corrected_text = text
        for error_phrase in sorted_error_phrases:
            correction = self.common_errors[error_phrase]
            
            # Build a regex pattern for the error phrase.
            # Handles spaces within the phrase and ensures whole word/phrase boundaries.
            # Case-insensitive matching.
            error_parts = [re.escape(part) for part in error_phrase.split(' ')]
            # For \b to work correctly with phrases, we need to be careful.
            # \b matches a word boundary. If error_phrase is "metform in", pattern is r'\bmetform\s+in\b'
            # This should work for phrases where components are words.
            pattern_str = r'\b' + r'\s+'.join(error_parts) + r'\b'
            
            # Simple replacement function that attempts to preserve case of the first letter
            # This is a heuristic and might not cover all casing scenarios perfectly.
            def replace_match(matchobj):
                matched_text = matchobj.group(0)
                if matched_text.istitle() and len(correction.split()) == 1: # Capitalized and single word correction
                    return correction.capitalize()
                if matched_text.isupper() and len(correction.split()) == 1: # All caps and single word correction
                    return correction.upper()
                # For multi-word corrections or other cases, use dictionary casing
                return correction

            corrected_text = re.sub(pattern_str, replace_match, corrected_text, flags=re.IGNORECASE)
            
        return corrected_text

    def correct_numbers(self, text: str) -> str:
        if not text: return text
            
        number_map = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen':'14',
            'fifteen':'15', 'sixteen':'16', 'seventeen':'17', 'eighteen':'18', 'nineteen':'19',
            'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50',
            'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
        }
        # Simpler replacement for standalone numbers or numbers before units
        for num_word, digit in number_map.items():
            text = re.sub(fr'\b{num_word}\b(?=\s+(mg|milligram|mcg|microgram|ml|milliliter|g|gram|percent|point|\d)|$)',
                          digit, text, flags=re.IGNORECASE)
        
        # Handle "X hundred"
        text = re.sub(r'\b(one|two|three|four|five|six|seven|eight|nine)\s+hundred\b', 
                      lambda m: f"{number_map.get(m.group(1).lower(), m.group(1))}00", text, flags=re.IGNORECASE)
        
        # Standardize dosage units (e.g., "milligrams" to "mg")
        text = re.sub(r'(\d+)\s*milligrams?\b', r'\1 mg', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*micrograms?\b', r'\1 mcg', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*milliliters?\b', r'\1 ml', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*grams?\b', r'\1 g', text, flags=re.IGNORECASE)
        
        # Ensure space between number and unit (e.g., "10mg" to "10 mg")
        text = re.sub(r'(\d)(mg|mcg|ml|g)\b', r'\1 \2', text, flags=re.IGNORECASE)
        
        # Percentage formatting
        text = re.sub(r'(\d+)\s*(percent|per cent|pct)\b', r'\1%', text, flags=re.IGNORECASE)
        # "point five" -> ".5" (when followed by a unit or standalone)
        text = re.sub(r'\bpoint\s+([a-zA-Z]+)\b', lambda m: f".{number_map.get(m.group(1).lower(), m.group(1))}" if m.group(1).lower() in number_map else f"point {m.group(1)}", text, flags=re.IGNORECASE)


        return text

    def correct_medical_terms(self, text: str) -> str:
        if not text: return text
        
        words = text.split()
        corrected_words = []
        all_medical_terms_flat = [term for sublist in self.medical_terms.values() for term in sublist]
        
        i = 0
        while i < len(words):
            # Consider n-grams of length 1 to 3 (or min(3, remaining words))
            max_ngram_len = min(3, len(words) - i)
            best_match_info = None # (corrected_term, score, ngram_len)

            for length in range(max_ngram_len, 0, -1): # Try longer n-grams first
                original_ngram_list = words[i : i + length]
                # Preserve original casing and punctuation for potential reapplication
                # For matching, use a cleaned version
                cleaned_ngram = " ".join(
                    word.lower().strip('.,;:!?()"\'') for word in original_ngram_list
                )

                if not cleaned_ngram or len(cleaned_ngram) <= 2: # Skip very short/empty ngrams
                    continue

                # Only perform fuzzy matching if it has some medical characteristics or is moderately long
                if len(cleaned_ngram) > 4 or self._has_medical_characteristics(cleaned_ngram):
                    # Fuzzy match against all medical terms
                    # process.extractOne returns (match, score)
                    match_candidate = process.extractOne(cleaned_ngram, all_medical_terms_flat, scorer=fuzz.WRatio)
                    
                    if match_candidate and match_candidate[1] >= 88: # Threshold for WRatio
                        # If this match is better than a previous shorter one for this position
                        if best_match_info is None or match_candidate[1] > best_match_info[1]:
                             best_match_info = (match_candidate[0], match_candidate[1], length)
            
            if best_match_info:
                corrected_term, score, ngram_len = best_match_info
                original_matched_phrase_list = words[i : i + ngram_len]
                
                # Attempt to preserve original casing (e.g. if first word was capitalized)
                if original_matched_phrase_list[0].istitle():
                    corrected_term = corrected_term.capitalize()
                elif original_matched_phrase_list[0].isupper():
                    # crude check for ALL CAPS for single word original, apply to single word correction
                    if len(original_matched_phrase_list) == 1 and original_matched_phrase_list[0].isupper() and ' ' not in corrected_term:
                         corrected_term = corrected_term.upper()

                # Re-attach punctuation from the last word of the original n-gram
                # This is a simplification; complex punctuation might not be perfectly preserved.
                last_word_original = original_matched_phrase_list[-1]
                leading_chars_last_word = last_word_original.rstrip('.,;:!?()"\'')
                trailing_punctuation = last_word_original[len(leading_chars_last_word):]
                
                corrected_words.append(corrected_term + trailing_punctuation)
                i += ngram_len
            else:
                corrected_words.append(words[i])
                i += 1
        
        return " ".join(corrected_words)

    def _has_medical_characteristics(self, text: str) -> bool:
        medical_suffixes = ["itis", "osis", "emia", "opathy", "ectomy", "pathy", "lysis", 
                           "trophy", "plasia", "oma", "ion", "ic", "al", "ive", "ary", "ine", "azole"]
        medical_prefixes = ["hyper", "hypo", "brady", "tachy", "dys", "poly", "hemi", 
                           "anti", "endo", "exo", "inter", "intra", "peri", "sub", "trans", "erythro"]
        
        text_lower = text.lower()
        for suffix in medical_suffixes:
            if text_lower.endswith(suffix): return True
        for prefix in medical_prefixes:
            if text_lower.startswith(prefix): return True
        # Check if any part of a multi-word phrase matches
        if ' ' in text_lower:
            if any(self._has_medical_characteristics(part) for part in text_lower.split()):
                return True
        return False

    def _is_valid_correction(self, original_text: str, corrected_text: str) -> bool:
        if not corrected_text.strip(): # Empty correction is invalid
            return False
        
        # Too drastic length change (e.g. > 3x shorter or longer) is suspicious
        # unless original text is very short
        len_orig = len(original_text)
        len_corr = len(corrected_text)
        if len_orig > 10 and (len_corr < len_orig / 3 or len_corr > len_orig * 3):
            # print(f"LLM Rejected: Drastic length change. Original: '{original_text}', Corrected: '{corrected_text}'")
            return False
        
        # Use FuzzyWuzzy's ratio. Threshold of 35 allows significant changes like "airy thameters" -> "erythematous"
        # fuzz.ratio("airy thameters", "erythematous") is 39
        # fuzz.ratio("metform in", "metformin") is 90
        similarity_score = fuzz.ratio(original_text.lower(), corrected_text.lower())
        
        if similarity_score < 35:
            # print(f"LLM Rejected: Low similarity ({similarity_score}). Original: '{original_text}', Corrected: '{corrected_text}'")
            return False
            
        # Check for repetitive placeholder/error phrases from LLM
        if corrected_text.lower() in ["corrected text:", "original text:", text.lower()]:
            return False
            
        return True
    
    def process_text(self, text: str, speaker_role: Optional[str] = None) -> str:
        """Process a single text segment with all correction methods (helper for external use)"""
        if not text or not text.strip():
            return text
            
        corrected = self.correct_common_errors(text)
        corrected = self.correct_numbers(corrected)
        corrected = self.correct_medical_terms(corrected)
        
        if self.use_llm and self.llm_model and self.llm_tokenizer:
            llm_candidate = self.correct_with_llm(corrected, speaker_role)
            if llm_candidate != corrected and self._is_valid_correction(corrected, llm_candidate):
                corrected = llm_candidate
            
        return corrected
    
def main():
    """Main function to run the error corrector on a transcript file"""
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python error_mitigation.py <transcript_json_file>")
        return
        
    # Get the transcript file from command line argument
    transcript_file = sys.argv[1]
    print(f"Processing transcript file: {transcript_file}")
    
    try:
        # Load the transcript data
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
            
        # Initialize the error corrector
        corrector = TranscriptionErrorCorrector(use_llm=True)
        
        # Process the transcript
        corrected_data = corrector.correct_transcript(transcript_data)
        
        # Write the corrected transcript to a new file
        output_file = transcript_file.replace('.json', '_corrected.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(corrected_data, f, indent=2, ensure_ascii=False)
            
        print(f"Corrected transcript written to: {output_file}")
        
        # Print some sample corrections if any were made
        if corrected_data.get("correction_stats", {}).get("total_turns_corrected", 0) > 0:
            print("\nSample corrections:")
            for turn in corrected_data.get("speaker_turns", []):
                if "original_text" in turn:
                    print(f"Original: {turn['original_text']}")
                    print(f"Corrected: {turn['text']}")
                    print("---")
                    break  # Just show one example
    
    except Exception as e:
        print(f"Error processing transcript: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()