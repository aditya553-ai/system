import os
import re
import json
from typing import Dict, List, Any, Optional
import Levenshtein
from nltk.tokenize import word_tokenize
import difflib
from collections import Counter
try:
    from fuzzywuzzy import process
    FUZZY_AVAILABLE = True
except ImportError:
    print("fuzzywuzzy not available. Some fuzzy matching features will be disabled.")
    FUZZY_AVAILABLE = False

# Import for LLM-based correction
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    import torch
    LLM_AVAILABLE = True
except ImportError:
    print("transformers or torch not available. LLM-based correction will be disabled.")
    LLM_AVAILABLE = False

class TranscriptionErrorCorrector:
    """
    Class for detecting and correcting errors in medical transcriptions
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize the error corrector
        
        Args:
            use_llm: Whether to use LLM for corrections
        """
        self.use_llm = use_llm and LLM_AVAILABLE
        self.llm_model = None
        self.llm_tokenizer = None
        self.model_type = None
        
        # Initialize LLM if enabled
        if self.use_llm:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM for error correction"""
        try:
            print("Loading LLM for medical transcription correction...")
            
            # Flexible model selection
            model_name = "google/flan-t5-base"  # Default model
            
            import threading
            import time
            import os
            
            # Set cache directory
            os.makedirs("model_cache", exist_ok=True)
            cache_dir = os.path.join(os.getcwd(), "model_cache")
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            print(f"Using model cache: {cache_dir}")
            
            load_success = [False]
            load_error = [None]
            
            def load_model():
                try:
                    # Load tokenizer
                    print(f"Loading tokenizer for {model_name}...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        cache_dir=cache_dir,
                        local_files_only=False
                    )
                    
                    # Determine model type and load appropriate model
                    print(f"Loading model {model_name}...")
                    if any(x in model_name.lower() for x in ["t5", "bart", "pegasus"]):
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            cache_dir=cache_dir,
                            torch_dtype=torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            local_files_only=False
                        )
                        model_type = "seq2seq"
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            cache_dir=cache_dir,
                            torch_dtype=torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            local_files_only=False
                        )
                        model_type = "causal"
                    
                    self.llm_tokenizer = tokenizer
                    self.llm_model = model
                    self.model_type = model_type
                    load_success[0] = True
                    print(f"Model {model_name} loaded successfully as {model_type} model!")
                except Exception as e:
                    load_error[0] = str(e)
                    print(f"Model loading error: {e}")
            
            # Start model loading in a separate thread
            load_thread = threading.Thread(target=load_model)
            load_thread.daemon = True
            load_thread.start()
            
            # Wait for the thread with a timeout
            timeout = 120
            start_time = time.time()
            while load_thread.is_alive() and time.time() - start_time < timeout:
                print(f"Still loading model... ({int(time.time() - start_time)}s)")
                time.sleep(5)
            
            if load_success[0]:
                print(f"LLM model '{model_name}' loaded successfully")
            elif load_error[0]:
                print(f"Failed to load LLM: {load_error[0]}")
                print("LLM-based correction will be disabled")
                self.use_llm = False
            else:
                print(f"LLM loading timed out after {timeout} seconds")
                self.use_llm = False
                
        except Exception as e:
            print(f"Failed to load LLM setup: {e}")
            self.use_llm = False

    def correct_with_llm(self, text: str, speaker_role: Optional[str] = None, medical_context: Optional[str] = None) -> str:
        """
        Correct transcription errors using LLM
        
        Args:
            text: Text to correct
            speaker_role: Role of speaker
            medical_context: Additional context
            
        Returns:
            Corrected text
        """
        if not self.use_llm or not self.llm_model:
            return text
        
        try:
            # Clean text for processing
            cleaned_text = text.strip('"\'')
            
            # Build prompt for correction
            role_context = f"Speaker is {speaker_role}" if speaker_role else ""
            medical_context_str = f"Previous context: {medical_context}" if medical_context else ""
            
            # Create prompt for medical term correction
            prompt = f"""Fix ONLY clearly misspelled medical terms in this text. Preserve all other text exactly.

**Rules:**
- Only correct obvious medication/medical condition misspellings
- Never change numbers, dates, or non-medical words
- Keep all punctuation and speaker annotations

Examples:
Input: "I took ibprophin for arthritus"
Output: "I took ibuprofen for arthritis"

Current Context: {medical_context_str}

Input Text: "{cleaned_text}"
Corrected Output:"""
            
            # Generate correction based on model type
            if self.model_type == "seq2seq":
                # Process for T5/BART models
                inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    self.llm_model = self.llm_model.to("cuda")
                
                # Conservative generation
                outputs = self.llm_model.generate(
                    **inputs,
                    max_length=len(cleaned_text) + 100,
                    min_length=max(10, len(cleaned_text) - 20),
                    temperature=0.1,
                    do_sample=False,
                    num_return_sequences=1
                )
                
                # Decode output
                generated_text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the correction - T5 may include the full prompt
                if "Corrected text" in generated_text:
                    corrected_text = generated_text.split("Corrected text")[-1].strip()
                    # Clean up the separator if present
                    corrected_text = corrected_text.strip(": (ONLY fix medical terms)").strip()
                else:
                    # Just take the last part of the output if no marker
                    corrected_text = generated_text.strip()
                
            else:
                # Process for causal LMs (GPT, Llama, etc.)
                inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True)
                
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    self.llm_model = self.llm_model.to("cuda")
                
                # Generate with stop tokens
                with torch.no_grad():
                    outputs = self.llm_model.generate(
                        **inputs,
                        max_new_tokens=len(cleaned_text) + 50,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.llm_tokenizer.eos_token_id
                    )
                
                # Decode the full output
                full_output = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract just the correction
                if "Corrected text" in full_output:
                    corrected_text = full_output.split("Corrected text (ONLY fix medical terms):")[-1].strip()
                else:
                    corrected_text = full_output.split("Original text:")[-1].strip()
                    corrected_text = corrected_text.replace(f'"{cleaned_text}"', '').strip()
            
            # Clean up any quotes
            corrected_text = corrected_text.strip('"\'')
            
            # Compare to original carefully using difflib
            similarity = difflib.SequenceMatcher(None, cleaned_text, corrected_text).ratio()
            
            # Reject if too different
            if similarity < 0.7:
                print(f"LLM made too many changes (similarity: {similarity:.2f}), using original")
                return text
                
            # Get specific changes for logging
            differ = difflib.Differ()
            diff = list(differ.compare(cleaned_text.split(), corrected_text.split()))
            changes = [(word.strip("+ ")) for word in diff if word.startswith("+ ")]
            
            # Log any actual changes
            if corrected_text != cleaned_text:
                print(f"LLM corrected: '{cleaned_text}' → '{corrected_text}'")
                if changes:
                    print(f"Changed words: {', '.join(changes)}")
                
                # Return corrected text
                return corrected_text
            else:
                # No changes needed
                return text
                
        except Exception as e:
            print(f"LLM correction failed: {e}")
            return text

    def correct_transcript(self, transcript_data: Dict) -> Dict:
        """
        Process a transcript with corrections
        
        Args:
            transcript_data: Transcript data
            
        Returns:
            Corrected transcript data
        """
        if not transcript_data or "speaker_turns" not in transcript_data:
            return transcript_data
            
        # Create deep copy to avoid modifying original
        import copy
        corrected_data = copy.deepcopy(transcript_data)
        corrections_made = 0
        
        print("Processing transcript for corrections...")
        
        # Process each turn
        for i, turn in enumerate(corrected_data.get("speaker_turns", [])):
            if "text" in turn:
                # Store the original text
                original_text = turn["text"]
                turn["original_text"] = original_text
                
                # Skip empty text
                if not original_text or len(original_text.strip()) == 0:
                    continue
                    
                # Get context from previous turns for better corrections
                context_turns = []
                for j in range(max(0, i-2), i):
                    prev_turn = corrected_data.get("speaker_turns", [])[j]
                    context_turns.append(f"{prev_turn.get('text', '')}")
                context = " ".join(context_turns)
                
                # Determine speaker role if possible
                speaker_id = turn.get("speaker", None)
                speaker_role = "doctor" if speaker_id == 0 else "patient" if speaker_id == 1 else None
                
                # Apply LLM correction with context awareness
                if self.use_llm and self.llm_model:
                    try:
                        corrected_text = self.correct_with_llm(
                            original_text, 
                            speaker_role=speaker_role,
                            medical_context=context
                        )
                        
                        # Record if changes were made
                        if corrected_text != original_text:
                            turn["text"] = corrected_text
                            corrections_made += 1
                            print(f"Turn {i+1} corrected: '{original_text}' → '{corrected_text}'")
                    except Exception as e:
                        print(f"Error correcting turn {i+1}: {e}")
        
        # Add correction statistics
        corrected_data["correction_stats"] = {
            "total_turns_corrected": corrections_made,
            "llm_corrections_applied": self.use_llm and self.llm_model is not None
        }
        
        print(f"Correction complete. {corrections_made} turns corrected.")
        
        return corrected_data
    
    def correct_numbers(self, text: str) -> str:
        """Correct numerical expressions in text"""
        # Convert spelled-out numbers to digits
        number_map = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50',
            'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
            'hundred': '100', 'thousand': '1000', 'million': '1000000', 'billion': '1000000000'
        }
        
        # Process common number patterns
        text = re.sub(r'\b(one|two|three|four|five) hundred\b', 
                      lambda m: f"{number_map[m.group(1)]}00", text)
                      
        # Handle other number words
        for word, digit in number_map.items():
            text = re.sub(r'\b' + word + r'\b', digit, text)
            
        return text
    
    def correct_medical_terms(self, text: str) -> str:
        """Correct common medical terms using fuzzy matching"""
        if not FUZZY_AVAILABLE:
            return text
            
        words = word_tokenize(text)
        corrected_words = []
        
        # Process each word
        for word in words:
            if len(word) <= 3:  # Skip very short words
                corrected_words.append(word)
                continue
                
            # Check for common misspellings
            if word.lower() in self.medical_terms['common_misspellings']:
                corrected_words.append(self.medical_terms['common_misspellings'][word.lower()])
                continue
                
            # Try fuzzy matching for longer words
            potential_matches = []
            for category in ['medications', 'conditions', 'procedures']:
                potential_matches.extend(self.medical_terms[category])
                
            # Only do fuzzy matching on longer words
            if len(word) >= 5:
                # Get best match
                match, score = process.extractOne(word.lower(), potential_matches)
                if score > 85:  # High confidence threshold
                    corrected_words.append(match)
                    continue
                    
            # No good match found
            corrected_words.append(word)
            
        return ' '.join(corrected_words)
    
    
