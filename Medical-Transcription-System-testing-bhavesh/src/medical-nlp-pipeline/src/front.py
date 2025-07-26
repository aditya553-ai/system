# streamlit_app.py
import os
import sys
import json
import time
import requests
import streamlit as st
from datetime import datetime
import tempfile
import io
import base64
import gc
import re

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

gc.collect()

# Add paths to access pipeline modules (if needed locally)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "medical-nlp-pipeline", "src"))

# Constants
API_URL = "http://localhost:8080"
PRESCRIPTION_API_URL = "http://localhost:8000"

# Set page config
st.set_page_config(
    page_title="Medical Transcription & Prescription System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add more robust error handling in the process_audio function
def process_audio(audio_file):
    """Process audio file through API to get SOAP note"""
    files = {"file": audio_file}
    data = {"wait_for_result": "true"}
    
    try:
        with st.spinner("Processing audio... This may take a few minutes."):
            try:
                response = requests.post(
                    f"{API_URL}/audio-to-soap",
                    files=files,
                    data=data,
                    timeout=600  # 10 minutes timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Store transcript content
                    if "transcript" in result:
                        # Ideal case: API provides structured speaker_turns
                        if "speaker_turns" in result["transcript"] and \
                           isinstance(result["transcript"]["speaker_turns"], list) and \
                           len(result["transcript"]["speaker_turns"]) > 0:
                            st.session_state.speaker_turns_transcript = result["transcript"]["speaker_turns"]
                            # For fallback or other uses, reconstruct a simple raw transcript
                            st.session_state.raw_transcript = "\n".join(
                                [turn.get("text", "") for turn in result["transcript"]["speaker_turns"]]
                            )
                        # Fallback: API provides only raw_text_for_nlp (or similar key)
                        elif "raw_text_for_nlp" in result["transcript"]: # Adjust key if API sends different
                            st.session_state.raw_transcript = result["transcript"]["raw_text_for_nlp"]
                        elif "raw" in result["transcript"]: # Older fallback
                             st.session_state.raw_transcript = result["transcript"]["raw"]
                    
                    return result
                else:
                    st.error(f"Error processing audio: {response.status_code}")
                    try:
                        error_detail = response.json()
                        st.error(f"Error details: {error_detail.get('message', 'Unknown error')}")
                    except:
                        st.error(response.text)
                    return None
            except requests.exceptions.ConnectionError as e:
                st.error(f"Connection error: Server may have crashed due to memory issues.")
                st.error("Try again with a shorter audio file or restart the server.")
                return None
            except requests.exceptions.ReadTimeout:
                st.error("Request timed out. The processing is taking too long.")
                return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_prescription(soap_note):
    """Get prescription recommendations from SOAP note"""
    # Extract patient data from SOAP note
    patient_data = {
        "patient_id": soap_note.get("patient_id", f"P{datetime.now().strftime('%Y%m%d%H%M%S')}"),
        "age": soap_note.get("age", 45),
        "gender": soap_note.get("gender", "Unknown"),
        "symptoms": [],
        "chronic_conditions": []
    }
    
    # Extract symptoms from SOAP note
    if "structured_data" in soap_note and "symptoms" in soap_note["structured_data"]:
        for symptom in soap_note["structured_data"]["symptoms"]:
            if "name" in symptom:
                patient_data["symptoms"].append(symptom["name"])
    
    # Extract chronic conditions
    if "structured_data" in soap_note and "problems" in soap_note["structured_data"]:
        for problem in soap_note["structured_data"]["problems"]:
            if "name" in problem:
                patient_data["chronic_conditions"].append(problem["name"])
    
    # If no symptoms found, try to extract from chief complaint
    if not patient_data["symptoms"] and "subjective" in soap_note:
        if "chief_complaint" in soap_note["subjective"]:
            chief = soap_note["subjective"]["chief_complaint"]
            if chief and chief != "Not assessed or documented":
                patient_data["symptoms"].append(chief)
    
    # If still no symptoms, add a default
    if not patient_data["symptoms"]:
        patient_data["symptoms"] = ["headache"]
    
    # Get prescription
    try:
        with st.spinner("Getting prescription recommendations..."):
            response = requests.post(
                f"{PRESCRIPTION_API_URL}/prescription/suggest",
                json=patient_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error getting prescription: {response.status_code}")
                st.error(response.text)
                return None
    except Exception as e:
        st.error(f"Error connecting to prescription API: {str(e)}")
        return None

def format_transcript_with_heuristic_speakers(raw_text: str) -> str:
    """
    Formats a raw transcript string with alternating speaker labels
    based on sentence segmentation. This is a heuristic.
    """
    if not raw_text or not isinstance(raw_text, str):
        return "Transcript not available or in unexpected format."
    
    # Basic sentence splitting using regex. Splits after '.', '?', '!' followed by whitespace.
    # This might not be perfect for all cases (e.g., "Dr. Smith" would be split).
    sentences = re.split(r'(?<=[.?!])\s+', raw_text.strip())
    
    formatted_lines = []
    # Assuming the conversation starts with the Patient, then Doctor, alternating.
    speakers = ["Patient", "Doctor"] 
    for i, sentence in enumerate(sentences):
        if sentence: # Avoid adding labels to empty strings from split
            speaker_label = speakers[i % 2]
            formatted_lines.append(f"{speaker_label}: {sentence}")
            
    return "\n\n".join(formatted_lines)

def display_soap_note(soap_note):
    """Display SOAP note in a structured format"""
    st.header("SOAP Note")
    
    # Display headers & patient info
    if "date" in soap_note:
        st.write(f"📅 Date: {soap_note['date']}")
    
    if "patient_id" in soap_note:
        st.write(f"🧑 Patient ID: {soap_note['patient_id']}")
    
    if "age" in soap_note:
        st.write(f"Age: {soap_note['age']}")
    
    if "gender" in soap_note:
        st.write(f"Gender: {soap_note['gender']}")
    
    # Subjective section
    with st.expander("SUBJECTIVE", expanded=True):
        if "subjective" in soap_note:
            subjective = soap_note["subjective"]
            
            if "chief_complaint" in subjective and subjective["chief_complaint"] != "Not assessed or documented":
                st.subheader("Chief Complaint")
                st.write(subjective["chief_complaint"])
            
            if "history_of_present_illness" in subjective and subjective["history_of_present_illness"] != "Not assessed or documented":
                st.subheader("History of Present Illness")
                hpi = subjective["history_of_present_illness"]
                if isinstance(hpi, dict) and "description" in hpi:
                    st.write(hpi["description"])
                else:
                    st.write(hpi)
            
            if "current_medications" in subjective and subjective["current_medications"] != "Not assessed or documented":
                st.subheader("Current Medications")
                st.write(subjective["current_medications"])
        else:
            st.write("No subjective information available")
    
    # Objective section
    with st.expander("OBJECTIVE", expanded=True):
        if "objective" in soap_note:
            objective = soap_note["objective"]
            
            if "vitals" in objective and objective["vitals"] != "Not assessed or documented":
                st.subheader("Vital Signs")
                st.write(objective["vitals"])
            
            if "physical_exam" in objective and objective["physical_exam"] != "Not assessed or documented":
                st.subheader("Physical Examination")
                st.write(objective["physical_exam"])
        else:
            st.write("No objective information available")
    
    # Assessment section
    with st.expander("ASSESSMENT", expanded=True):
        if "assessment" in soap_note:
            assessment = soap_note["assessment"]
            
            if isinstance(assessment, dict) and "diagnosis" in assessment:
                diagnosis = assessment["diagnosis"]
                
                if "primary_diagnosis" in diagnosis and diagnosis["primary_diagnosis"] != "Not assessed or documented":
                    st.subheader("Primary Diagnosis")
                    st.write(diagnosis["primary_diagnosis"])
                
                if "differential_diagnoses" in diagnosis and diagnosis["differential_diagnoses"] != "Not assessed or documented":
                    st.subheader("Differential Diagnoses")
                    st.write(diagnosis["differential_diagnoses"])
            elif isinstance(assessment, str) and assessment != "Not assessed or documented":
                st.write(assessment)
        else:
            st.write("No assessment information available")
    
    # Plan section
    with st.expander("PLAN", expanded=True):
        if "plan" in soap_note:
            plan = soap_note["plan"]
            
            if isinstance(plan, dict):
                if "current_medications" in plan and plan["current_medications"] != "Not assessed or documented":
                    st.subheader("Current Medications")
                    st.write(plan["current_medications"])
                
                if "new_prescriptions" in plan and plan["new_prescriptions"] != "Not assessed or documented":
                    st.subheader("New Prescriptions")
                    st.write(plan["new_prescriptions"])
                if "plan_text" in plan and plan["plan_text"] != "Not assessed or documented":
                    st.subheader("Treatment Plan")
                    st.write(plan["plan_text"])
            elif isinstance(plan, str):
                st.write(plan)
        else:
            st.write("No plan information available")
    
    # Medical Codes section - NEW ADDITION
    if "structured_data" in soap_note:
        with st.expander("MEDICAL CODES & STRUCTURED DATA", expanded=False):
            structured_data = soap_note["structured_data"]
            
            # Display symptoms with codes
            if "symptoms" in structured_data and structured_data["symptoms"]:
                st.subheader("🩺 Symptoms & Diagnoses")
                for i, symptom in enumerate(structured_data["symptoms"], 1):
                    with st.container():
                        col1, col2 = st.columns([2, 3])
                        
                        with col1:
                            st.markdown(f"**{i}. {symptom.get('name', 'Unknown')}")
                            if 'description' in symptom:
                                st.caption(symptom['description'])
                                
                        with col2:
                            codes_found = False
                            if symptom.get('icd10_code'):
                                st.code(f"ICD-10: {symptom['icd10_code']}", language=None)
                                codes_found = True
                            if symptom.get('snomed_code'):
                                st.code(f"SNOMED: {symptom['snomed_code']}", language=None)
                                codes_found = True
                            if symptom.get('codes') and symptom['codes']:
                                st.code(f"Codes: {symptom['codes']}", language=None)
                                codes_found = True
                            if not codes_found:
                                st.caption("No medical codes available")
                        
                        if 'confidence' in symptom:
                            st.caption(f"Confidence: {symptom['confidence']}")
                        st.divider()
            
            # Display medications with codes
            if "medications" in structured_data and structured_data["medications"]:
                st.subheader("💊 Medications")
                for i, med in enumerate(structured_data["medications"], 1):
                    with st.container():
                        col1, col2 = st.columns([2, 3])
                        
                        with col1:
                            st.markdown(f"**{i}. {med.get('name', 'Unknown')}")
                            if 'dose' in med:
                                st.caption(f"Dose: {med['dose']}")
                            if 'frequency' in med:
                                st.caption(f"Frequency: {med['frequency']}")
                                
                        with col2:
                            codes_found = False
                            if med.get('rxnorm_code'):
                                st.code(f"RxNorm: {med['rxnorm_code']}", language=None)
                                codes_found = True
                            if med.get('codes') and med['codes']:
                                st.code(f"Codes: {med['codes']}", language=None)
                                codes_found = True
                            if not codes_found:
                                st.caption("No medication codes available")
                        
                        if 'confidence' in med:
                            st.caption(f"Confidence: {med['confidence']}")
                        st.divider()
            
            # Display current medications if different from medications
            if "current_medications" in structured_data and structured_data["current_medications"]:
                current_meds = structured_data["current_medications"]
                # Only show if different from medications section
                if current_meds != structured_data.get("medications", []):
                    st.subheader("🔄 Current Medications")
                    for i, med in enumerate(current_meds, 1):
                        with st.container():
                            col1, col2 = st.columns([2, 3])
                            
                            with col1:
                                st.markdown(f"**{i}. {med.get('name', 'Unknown')}")
                                if 'dose' in med:
                                    st.caption(f"Dose: {med['dose']}")
                                if 'frequency' in med:
                                    st.caption(f"Frequency: {med['frequency']}")
                                    
                            with col2:
                                codes_found = False
                                if med.get('rxnorm_code'):
                                    st.code(f"RxNorm: {med['rxnorm_code']}", language=None)
                                    codes_found = True
                                if med.get('codes') and med['codes']:
                                    st.code(f"Codes: {med['codes']}", language=None)
                                    codes_found = True
                                if not codes_found:
                                    st.caption("No medication codes available")
                            
                            if 'confidence' in med:
                                st.caption(f"Confidence: {med['confidence']}")
                            st.divider()
            
            # Display other structured data
            other_keys = [key for key in structured_data.keys() 
                         if key not in ['symptoms', 'medications', 'current_medications']]
            
            if other_keys:
                st.subheader("📋 Additional Structured Data")
                for key in other_keys:
                    value = structured_data[key]
                    if value:  # Only show non-empty values
                        st.markdown(f"**{key.replace('_', ' ').title()}:**")
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict):
                                    st.json(item)
                                else:
                                    st.write(f"• {item}")
                        elif isinstance(value, dict):
                            st.json(value)
                        else:
                            st.write(value)
                        st.write("")
            
            # Summary section
            st.subheader("📊 Extraction Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                symptom_count = len(structured_data.get("symptoms", []))
                st.metric("Symptoms Extracted", symptom_count)
                
            with summary_col2:
                med_count = len(structured_data.get("medications", []))
                st.metric("Medications Extracted", med_count)
                
            with summary_col3:
                total_codes = 0
                for symptom in structured_data.get("symptoms", []):
                    if symptom.get('icd10_code') or symptom.get('snomed_code') or symptom.get('codes'):
                        total_codes += 1
                for med in structured_data.get("medications", []):
                    if med.get('rxnorm_code') or med.get('codes'):
                        total_codes += 1
                st.metric("Items with Codes", total_codes)
    
    # Alerts section
    if "alerts" in soap_note and soap_note["alerts"]:
        with st.expander("ALERTS"):
            for alert in soap_note["alerts"]:
                st.warning(alert)

def display_prescription(prescription):
    """Display prescription recommendations"""
    st.header("💊 Prescription Recommendations")
    
    if "diagnosis" in prescription:
        st.subheader("🩺 Diagnosis")
        st.write(prescription["diagnosis"])
    
    if "medicines" in prescription:
        st.subheader("💉 Recommended Medications")
        for i, medicine in enumerate(prescription["medicines"]):
            with st.container():
                med_col1, med_col2 = st.columns([3, 2])
                
                with med_col1:
                    st.markdown(f"**{i+1}. {medicine.get('medicineName', 'Unknown')} {medicine.get('dosage', '')}**")
                    st.write(f"📅 Frequency: {medicine.get('frequency', 'As directed')}")
                    st.write(f"⏱️ Duration: {medicine.get('duration', 'As needed')} days")
                    st.write(f"📋 Instructions: {medicine.get('instructions', 'Take as directed')}")
                
                with med_col2:
                    # Display medical codes section
                    st.markdown("**🏷️ Medical Codes:**")
                    codes_found = False
                    
                    if medicine.get('rxnorm_code'):
                        st.code(f"RxNorm: {medicine['rxnorm_code']}", language=None)
                        codes_found = True
                    if medicine.get('ndc_code'):
                        st.code(f"NDC: {medicine['ndc_code']}", language=None)
                        codes_found = True
                    if medicine.get('codes'):
                        st.code(f"Codes: {medicine['codes']}", language=None)
                        codes_found = True
                    
                    if not codes_found:
                        st.caption("No medical codes available")
                    
                    # Display additional medication info if available
                    if medicine.get('strength'):
                        st.caption(f"💪 Strength: {medicine['strength']}")
                    if medicine.get('route'):
                        st.caption(f"📍 Route: {medicine['route']}")
                    if medicine.get('confidence'):
                        st.caption(f"📊 Confidence: {medicine['confidence']}")
                
                # Chemical composition in expandable section
                if "chemicalComposition" in medicine and medicine["chemicalComposition"]:
                    with st.expander("🧪 Chemical Composition"):
                        st.write(medicine["chemicalComposition"])
                
                st.divider()
    
    if "doctorAdvice" in prescription:
        st.subheader("👨‍⚕️ Doctor's Advice")
        st.write(prescription["doctorAdvice"])
    
    if "followUpDate" in prescription:
        st.subheader("📅 Follow-up")
        st.write(f"Recommended follow-up: {prescription['followUpDate']}")
    
    # Enhanced drug interactions display
    if "interactions" in prescription and prescription["interactions"]:
        with st.expander("⚠️ Potential Drug Interactions", expanded=True):
            if isinstance(prescription["interactions"], list):
                for interaction in prescription["interactions"]:
                    if isinstance(interaction, dict):
                        severity = interaction.get('severity', 0)
                        severity_color = "🔴" if severity > 0.7 else "🟠" if severity > 0.3 else "🟡"
                        
                        st.warning(
                            f"{severity_color} **Interaction between {interaction.get('med1', 'Unknown')} and {interaction.get('med2', 'Unknown')}**\n\n"
                            f"**Severity Score**: {severity:.2f}\n\n"
                            f"**Category**: {interaction.get('category', 'Unknown')}\n\n"
                            f"**Effect**: {interaction.get('effect', 'No details available')}"
                        )
                    else:
                        st.write(f"• {interaction}")
            else:
                st.write(prescription["interactions"])
    
    # Prescription metadata
    if "prescription_id" in prescription:
        st.caption(f"Prescription ID: {prescription['prescription_id']}")
    
    # Download prescription as JSON
    prescription_json = json.dumps(prescription, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 Download Prescription (JSON)",
        data=prescription_json,
        file_name=f"prescription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def render_html_button(html_content, button_text="View HTML"):
    """Create a button to view HTML content in a new tab"""
    html_b64 = base64.b64encode(html_content.encode()).decode()
    href = f'data:text/html;base64,{html_b64}'
    return st.markdown(f'<a href="{href}" target="_blank"><button>{button_text}</button></a>', unsafe_allow_html=True)

# Main Streamlit app
def main():
    gc.collect()
    st.title("Medical Transcription & Prescription System")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Upload Audio", "View Results"])
    
    # Session state initialization
    if "soap_note" not in st.session_state:
        st.session_state.soap_note = None
    if "prescription" not in st.session_state:
        st.session_state.prescription = None
    if "raw_transcript" not in st.session_state:
        st.session_state.raw_transcript = None
    # Change from diarized_transcript to speaker_turns_transcript for clarity
    if "speaker_turns_transcript" not in st.session_state: 
        st.session_state.speaker_turns_transcript = None
    
    # Page logic
    if page == "Upload Audio":
        st.header("Upload Medical Audio Recording")
        
        uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg"])
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            if st.button("Process Audio"):
                # Reset previous results
                st.session_state.soap_note = None
                st.session_state.prescription = None
                st.session_state.raw_transcript = None
                st.session_state.speaker_turns_transcript = None

                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name
                
                try:
                    result = process_audio(uploaded_file) # process_audio now updates session states
                    
                    if result and "soap_note" in result:
                        st.session_state.soap_note = result["soap_note"]
                        st.success("SOAP note generated successfully!")
                        
                        # Transcripts are now set within process_audio or from files if paths are returned
                        if "files" in result: # Handling file paths if API returns them
                            if "raw_transcript" in result["files"] and not st.session_state.raw_transcript:
                                try:
                                    with open(result["files"]["raw_transcript"], 'r', encoding='utf-8') as f:
                                        st.session_state.raw_transcript = f.read()
                                except Exception as e:
                                    st.warning(f"Could not read raw transcript file: {e}")
                            
                            
                        
                        st.rerun() # Rerun to navigate or update view
                    elif result: # API returned something, but not the expected SOAP note
                        st.error("Processing completed, but failed to retrieve SOAP note structure.")
                        if st.session_state.raw_transcript:
                             st.info("Raw transcript was retrieved. Check 'View Results'.")
                             st.rerun()
                    else: # process_audio returned None (error already shown)
                        pass

                except Exception as e:
                    st.error(f"Error during audio processing workflow: {str(e)}")
                finally:
                    try:
                        if 'audio_path' in locals() and os.path.exists(audio_path):
                            os.unlink(audio_path)
                    except Exception as e:
                        st.warning(f"Could not delete temp file {audio_path}: {e}")
    
    elif page == "View Results":
        if st.session_state.soap_note or st.session_state.raw_transcript or st.session_state.speaker_turns_transcript:
            tab1, tab2, tab3, tab4 = st.tabs(["SOAP Note", "Transcripts", "Prescription", "Raw Data & Debug"])
            
            with tab1:
                if st.session_state.soap_note:
                    display_soap_note(st.session_state.soap_note)
                    soap_json = json.dumps(st.session_state.soap_note, indent=2)
                    st.download_button(
                        label="Download SOAP Note (JSON)",
                        data=soap_json,
                        file_name="soap_note.json",
                        mime="application/json"
                    )
                else:
                    st.info("SOAP note not available.")
            
            with tab2: # Transcripts Tab
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Raw Transcript")
                    if st.session_state.speaker_turns_transcript:
                        st.markdown("_(Displaying ASR-provided speaker turns)_")
                        transcript_display_text = ""
                        # Define speaker mapping (can be made more dynamic if needed)
                        speaker_map = {0: "Patient", 1: "Doctor"} 
                        # If num_speakers from metadata is available and > 2, adjust map or use generic labels
                        
                        for turn in st.session_state.speaker_turns_transcript:
                            speaker_id = turn.get("speaker")
                            text = turn.get("text", "")
                            # Use mapped name if available, else generic "Speaker X"
                            speaker_label = speaker_map.get(speaker_id, f"Speaker {speaker_id}")
                            transcript_display_text += f"**{speaker_label}:** {text}\n\n"
                        st.markdown(transcript_display_text, unsafe_allow_html=True) # Using markdown for bold
                    elif st.session_state.raw_transcript:
                        st.markdown("_(Displaying raw transcript with heuristic speaker labels - ASR speaker turns not available)_")
                        formatted_text = format_transcript_with_heuristic_speakers(st.session_state.raw_transcript)
                        st.text_area("Formatted Raw Transcript", formatted_text, height=400, key="formatted_raw_transcript_heuristic")
                    else:
                        st.info("Raw transcript not available")
                
            with tab3:
                if st.session_state.prescription:
                    display_prescription(st.session_state.prescription)
                elif st.session_state.soap_note: # Only show button if SOAP note exists
                    st.info("No prescription recommendations available yet.")
                    if st.button("Generate Prescription Recommendations"):
                        prescription = get_prescription(st.session_state.soap_note)
                        if prescription:
                            st.session_state.prescription = prescription
                            st.success("Prescription recommendations generated!")
                            st.rerun()
                        else:
                            st.error("Failed to generate prescription recommendations.")
                else:
                    st.info("Generate a SOAP note first to enable prescription recommendations.")
            
            with tab4:  # Raw Data & Debug Tab
                st.header("🔧 Raw Data & Debug Information")
                
                if st.session_state.soap_note:
                    # Full SOAP Note JSON
                    with st.expander("📄 Complete SOAP Note (JSON)", expanded=False):
                        st.json(st.session_state.soap_note)
                        
                        # Download button for raw JSON
                        soap_json = json.dumps(st.session_state.soap_note, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📥 Download Raw SOAP JSON",
                            data=soap_json,
                            file_name=f"soap_note_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    # Structured Data Analysis
                    if "structured_data" in st.session_state.soap_note:
                        with st.expander("🧬 Structured Data Analysis", expanded=True):
                            structured_data = st.session_state.soap_note["structured_data"]
                            
                            st.subheader("Data Structure Overview")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Keys", len(structured_data.keys()))
                            with col2:
                                total_entities = 0
                                for key, value in structured_data.items():
                                    if isinstance(value, list):
                                        total_entities += len(value)
                                st.metric("Total Entities", total_entities)
                            with col3:
                                coded_entities = 0
                                for key, value in structured_data.items():
                                    if isinstance(value, list):
                                        for item in value:
                                            if isinstance(item, dict):
                                                for k in item.keys():
                                                    if k.endswith('_code') and item[k]:
                                                        coded_entities += 1
                                                        break
                                st.metric("Entities with Codes", coded_entities)
                            
                            st.subheader("Raw Structured Data")
                            st.json(structured_data)
                    
                    # System Information
                    with st.expander("🖥️ System Information", expanded=False):
                        st.subheader("Processing Metadata")
                        
                        # Display processing date/time if available
                        if "date" in st.session_state.soap_note:
                            st.write(f"**Processing Date:** {st.session_state.soap_note['date']}")
                        
                        if "patient_id" in st.session_state.soap_note:
                            st.write(f"**Patient ID:** {st.session_state.soap_note['patient_id']}")
                        
                        # API Status Check
                        st.subheader("API Status")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("🔍 Test Medical API"):
                                try:
                                    response = requests.get(f"{API_URL}/health", timeout=5)
                                    if response.status_code == 200:
                                        st.success("✅ Medical API is responding")
                                    else:
                                        st.error(f"❌ Medical API error: {response.status_code}")
                                except Exception as e:
                                    st.error(f"❌ Medical API unreachable: {str(e)}")
                        
                        with col2:
                            if st.button("🏥 Test Prescription API"):
                                try:
                                    response = requests.get(f"{PRESCRIPTION_API_URL}/health", timeout=5)
                                    if response.status_code == 200:
                                        st.success("✅ Prescription API is responding")
                                    else:
                                        st.error(f"❌ Prescription API error: {response.status_code}")
                                except Exception as e:
                                    st.error(f"❌ Prescription API unreachable: {str(e)}")
                
                # Transcript debugging
                if st.session_state.speaker_turns_transcript or st.session_state.raw_transcript:
                    with st.expander("🎤 Transcript Data Analysis", expanded=False):
                        if st.session_state.speaker_turns_transcript:
                            st.subheader("Speaker-Labeled Transcript")
                            st.write(f"**Total Turns:** {len(st.session_state.speaker_turns_transcript)}")
                            
                            # Speaker analysis
                            speakers = set()
                            turn_counts = {}
                            for turn in st.session_state.speaker_turns_transcript:
                                speaker = turn.get("speaker", "Unknown")
                                speakers.add(speaker)
                                turn_counts[speaker] = turn_counts.get(speaker, 0) + 1
                            
                            st.write(f"**Unique Speakers:** {len(speakers)}")
                            st.write("**Turn Distribution:**")
                            for speaker, count in turn_counts.items():
                                st.write(f"  - Speaker {speaker}: {count} turns")
                            
                            st.subheader("Raw Speaker Turns Data")
                            st.json(st.session_state.speaker_turns_transcript)
                        
                        if st.session_state.raw_transcript:
                            st.subheader("Raw Transcript")
                            st.text_area("Raw Transcript Content", st.session_state.raw_transcript, height=200)
                
                if st.session_state.prescription:
                    with st.expander("💊 Prescription Debug Data", expanded=False):
                        st.subheader("Raw Prescription Response")
                        st.json(st.session_state.prescription)
                
                # Session State Debugging
                with st.expander("💾 Session State Debug", expanded=False):
                    st.subheader("Current Session Variables")
                    session_info = {
                        "soap_note_available": bool(st.session_state.soap_note),
                        "raw_transcript_available": bool(st.session_state.raw_transcript),
                        "speaker_turns_available": bool(st.session_state.speaker_turns_transcript),
                        "prescription_available": bool(st.session_state.prescription),
                    }
                    
                    if st.session_state.soap_note:
                        session_info["soap_note_keys"] = list(st.session_state.soap_note.keys())
                    
                    st.json(session_info)
                    
                    # Clear session button
                    if st.button("🗑️ Clear All Session Data", help="Clear all cached data and reset the session"):
                        for key in ["soap_note", "raw_transcript", "speaker_turns_transcript", "prescription"]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.success("Session data cleared! Please refresh the page.")
                        st.rerun()

        else:
            st.info("No results available. Please upload an audio file first on the 'Upload Audio' page.")
            
            # Demo button for testing
            if st.button("Load Demo SOAP Note"):
                # Create a simple demo SOAP note
                st.session_state.soap_note = {
                    "patient_id": "P20230405",
                    "age": 45,
                    "gender": "Female",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "subjective": {
                        "chief_complaint": "Severe headache",
                        "history_of_present_illness": "Patient reports throbbing headache for 3 days, associated with nausea and sensitivity to light.",
                        "current_medications": "Ibuprofen 400mg PRN"
                    },
                    "objective": {
                        "vitals": "BP 128/82, HR 76, Temp 98.6F",
                        "physical_exam": "No focal neurological deficits. Mild tenderness in temporal region."
                    },
                    "assessment": {
                        "diagnosis": {
                            "primary_diagnosis": "Migraine without aura",
                            "differential_diagnoses": "Tension headache, Cluster headache, Sinusitis"
                        }
                    },
                    "plan": {
                        "current_medications": "Continue Ibuprofen as needed",
                        "new_prescriptions": "Sumatriptan 50mg - Take 1 tablet at onset of headache, may repeat after 2 hours if needed",
                        "plan_text": "1. Start Sumatriptan for acute treatment\n2. Maintain headache diary\n3. Avoid known triggers\n4. Follow up in 2 weeks"
                    },
                    "alerts": [],
                    "structured_data": {
                        "symptoms": [
                            {"name": "headache", "description": "Severe throbbing headache"},
                            {"name": "nausea", "description": "Mild nausea"},
                            {"name": "photophobia", "description": "Sensitivity to light"}
                        ],
                        "problems": [
                            {"name": "Migraine"}
                        ]
                    }
                }
                st.session_state.speaker_turns_transcript = [
                    {"speaker": 0, "text": "Hi Doctor, I have a severe headache. It's been throbbing for 3 days. I also feel nauseous and sensitive to light. I've been taking Ibuprofen."},
                    {"speaker": 1, "text": "Okay, let's check your vitals. Your blood pressure is 128/82, heart rate 76, temperature 98.6. Any neurological issues? Tenderness?"},
                    {"speaker": 0, "text": "No major issues, just some tenderness in my temples."},
                    {"speaker": 1, "text": "Based on this, it looks like a migraine without aura. We could also consider tension headache or sinusitis. For now, continue Ibuprofen. I'll also prescribe Sumatriptan 50mg. Take one at onset, and you can repeat in 2 hours if needed. Keep a headache diary, avoid triggers, and follow up in 2 weeks."}
                ]
                # For demo purposes, also create a sample raw transcript
                st.session_state.raw_transcript = None # Clear this if using speaker_turns for demo
                st.success("Demo SOAP note and sample speaker-labeled transcript loaded!")
                st.rerun()


# Run the app
if __name__ == "__main__":
    main()