# api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import os
import sys
import json
import time
import uuid
import shutil
from datetime import datetime, timedelta
import uvicorn
import gc
import asyncio
import argparse
import logging
import psutil
import traceback
from model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)

def log_memory_usage(label=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    logging.info(f"Memory usage ({label}): {mem_mb:.2f} MB")
    return mem_mb

USE_FAST_RESOLVER = False  # Set to False to reduce memory usage
MAX_TRANSCRIPT_LENGTH = 10000  # Maximum transcript length to process

# Add paths to access pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "medical-nlp-pipeline", "src"))

ASR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "asr")
sys.path.insert(0, ASR_PATH)

# Import ASR module
try:
    from transcribe import transcribe_with_whisper
    ASR_AVAILABLE = True
except ImportError:
    print("Warning: ASR module not found. Audio transcription will not be available.")
    ASR_AVAILABLE = False


model_manager = ModelManager()
pipeline = model_manager.create_pipeline(use_fast_resolver=USE_FAST_RESOLVER)
# Initialize FastAPI app
app = FastAPI(
    title="Medical Transcription & Prescription API",
    description="API for processing medical transcriptions and suggesting prescriptions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure storage paths
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploads")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# Track processing jobs
jobs = {}

# --- Data Models ---

class TranscriptRequest(BaseModel):
    text: str = Field(..., description="The transcript text to process")
    job_name: Optional[str] = Field(None, description="Optional name for the job")

class AudioTranscriptionRequest(BaseModel):
    job_name: Optional[str] = Field(None, description="Optional name for the job")
    transcription_options: Optional[Dict[str, Any]] = Field(None, description="Options for transcription")

class PatientData(BaseModel):
    patient_id: str
    age: int = Field(..., ge=0, le=120)
    gender: str
    symptoms: List[str]
    chronic_conditions: Optional[List[str]] = []

class MedicineDetails(BaseModel):
    medicineName: str
    dosage: str
    frequency: str
    instructions: str
    duration: int
    chemicalComposition: Optional[str] = None

class PrescriptionResponse(BaseModel):
    prescription_id: str
    diagnosis: str
    medicines: List[MedicineDetails]
    doctorAdvice: str
    followUpDate: str
    interactions: Optional[List[Dict[str, Any]]] = []

class PrescriptionRequest(BaseModel):
    patient_data: PatientData
    soap_note_id: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    result_files: Optional[Dict[str, str]] = None

# --- Helper Functions ---

def generate_job_id():
    """Generate a unique job ID"""
    return str(uuid.uuid4())

def save_transcript(transcript_text, job_id):
    """Save transcript text to file"""
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    return file_path

def save_uploaded_file(upload_file, job_id):
    """Save uploaded file"""
    file_extension = os.path.splitext(upload_file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}{file_extension}")
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    
    return file_path

def process_transcript_task(job_id, transcript_path):
    """Background task to process transcript"""
    try:
        # Update job status
        jobs[job_id]["status"] = "processing"
        
        # Create output directory for this job
        job_output_dir = os.path.join(RESULTS_DIR, job_id)
        os.makedirs(job_output_dir, exist_ok=True)
        
        # Process the transcript
        result = pipeline.process_transcript(transcript_path, job_output_dir)
        
        # Update job status with results
        if result:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
            # Collect result file paths
            result_files = {}
            if "notes" in result:
                for format_name, file_path in result["notes"].items():
                    if file_path and os.path.exists(file_path):
                        result_files[format_name] = file_path
            
            jobs[job_id]["result_files"] = result_files
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = "Failed to process transcript"
    
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = str(e)
        import traceback
        traceback.print_exc()

def extract_patient_data_from_soap(soap_note_path):
    """Extract patient data from SOAP note for prescription"""
    try:
        with open(soap_note_path, 'r', encoding='utf-8') as f:
            soap_note = json.load(f)
        
        # Extract basic info
        patient_id = soap_note.get("patient_id", f"P{datetime.now().strftime('%Y%m%d')}")
        age = soap_note.get("age", 45)  # Default age if not found
        gender = soap_note.get("gender", "Unknown")
        
        # Extract symptoms
        symptoms = []
        if "structured_data" in soap_note and "symptoms" in soap_note["structured_data"]:
            for symptom in soap_note["structured_data"]["symptoms"]:
                if "name" in symptom:
                    symptoms.append(symptom["name"])

        # Extract chronic conditions
        chronic_conditions = []
        if "structured_data" in soap_note and "problems" in soap_note["structured_data"]:
            for problem in soap_note["structured_data"]["problems"]:
                if "name" in problem:
                    chronic_conditions.append(problem["name"])
        
        # If no symptoms found, look in the plan
        if not symptoms and "structured_data" in soap_note and "plan" in soap_note["structured_data"]:
            plan = soap_note["structured_data"]["plan"]
            symptom_keywords = ["headache", "migraine", "pain", "nausea", "dizziness"]
            for keyword in symptom_keywords:
                if keyword in plan.lower() and keyword not in symptoms:
                    symptoms.append(keyword)
        
        # Default symptom if none found
        if not symptoms:
            symptoms = ["headache"]
        
        return {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "chronic_conditions": chronic_conditions
        }
        
    except Exception as e:
        print(f"Error extracting patient data: {e}")
        return None

def get_prescription(patient_data, prescription_api_url="http://localhost:8000"):
    """Get prescription from external API"""
    import requests
    
    try:
        response = requests.post(
            f"{prescription_api_url}/prescription/suggest",
            json=patient_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error getting prescription: {e}")
        return None

# --- API Routes ---

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Welcome to the Medical Transcription & Prescription API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.post("/audio-to-soap")
async def audio_to_soap(
    file: UploadFile = File(...),
    wait_for_result: bool = Form(False),
    timeout_seconds: int = Form(300)  # 5 minutes default timeout
):
    """Process audio file directly to SOAP note"""
    if not ASR_AVAILABLE:
        raise HTTPException(status_code=400, detail="ASR module not available")
    
    # Generate job ID
    job_id = generate_job_id()
    logging.info(f"New job created: {job_id}")
    
    # Log memory at start
    log_memory_usage(f"Starting job {job_id}")
    
    # Save audio file
    audio_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, file.filename)
    
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Create result directory
    result_dir = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(result_dir, exist_ok=True)
    
    # Create job entry
    jobs[job_id] = {
        "job_id": job_id,
        "name": f"Audio-to-SOAP: {file.filename}",
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "audio_path": audio_path,
        "result_dir": result_dir
    }
    
    # Process in background or wait
    if wait_for_result:
        try:
            # Process directly
            jobs[job_id]["status"] = "transcribing"
            jobs[job_id]["message"] = "Transcribing audio..."
            
            # Force garbage collection
            gc.collect()
            
            # Step 1: Transcribe with reduced memory usage
            logging.info(f"Job {job_id}: Starting transcription")
            try:
                transcript_text = transcribe_with_whisper(audio_path)
                if isinstance(transcript_text, dict) and "text" in transcript_text:
                    transcript_text = transcript_text["text"]
                    
                # Limit transcript length if needed
                if len(transcript_text) > MAX_TRANSCRIPT_LENGTH:
                    logging.warning(f"Transcript too long ({len(transcript_text)} chars), truncating")
                    transcript_text = transcript_text[:MAX_TRANSCRIPT_LENGTH]
                
                # Save transcript
                transcript_path = os.path.join(result_dir, "transcript.txt")
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(str(transcript_text))
                
                # Store raw transcript text
                raw_transcript = str(transcript_text)
                jobs[job_id]["raw_transcript"] = raw_transcript
                
            except Exception as e:
                logging.error(f"Error in transcription: {str(e)}")
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["message"] = f"Transcription error: {str(e)}"
                
                return JSONResponse(status_code=500, content={
                    "job_id": job_id,
                    "status": "failed",
                    "message": f"Transcription error: {str(e)}"
                })
            
            # Log memory after transcription
            log_memory_usage(f"After transcription for job {job_id}")
            
            # Force garbage collection
            gc.collect()
            
            # Step 2: Process transcript
            jobs[job_id]["status"] = "processing"
            jobs[job_id]["message"] = "Processing transcript..."
            
            try:
                # Process transcript
                logging.info(f"Job {job_id}: Starting NLP processing")
                result = pipeline.process_transcript(transcript_text, result_dir)
                
            except Exception as e:
                logging.error(f"Error in NLP processing: {str(e)}")
                traceback.print_exc()
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["message"] = f"NLP processing error: {str(e)}"
                
                return JSONResponse(status_code=500, content={
                    "job_id": job_id,
                    "status": "failed",
                    "message": f"NLP processing error: {str(e)}",
                    "transcript": {
                        "raw": jobs[job_id].get("raw_transcript", "")
                    }
                })
            
            # Log memory after processing
            log_memory_usage(f"After NLP processing for job {job_id}")
            
            # Step 3: Handle results
            try:
                # Update job status
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["completed_at"] = datetime.now().isoformat()
                
                # Collect result files
                result_files = {}
                if result and "notes" in result:
                    for format_name, file_path in result["notes"].items():
                        if file_path and os.path.exists(file_path):
                            result_files[format_name] = file_path
                
                jobs[job_id]["result_files"] = result_files
                
                # Return SOAP note JSON
                soap_path = result_files.get("json")
                if soap_path and os.path.exists(soap_path):
                    try:
                        with open(soap_path, 'r', encoding='utf-8') as f:
                            soap_data = json.load(f)
                        
                        return {
                            "job_id": job_id,
                            "status": "completed",
                            "soap_note": soap_data,
                            "transcript": {
                                "raw": jobs[job_id].get("raw_transcript", "")
                            },
                            "completed_at": jobs[job_id].get("completed_at"),
                            "files": result_files
                        }
                    except Exception as e:
                        logging.error(f"Error reading SOAP note: {str(e)}")
                        return {
                            "job_id": job_id,
                            "status": "completed",
                            "message": "SOAP note generated but error reading JSON",
                            "transcript": {
                                "raw": jobs[job_id].get("raw_transcript", "")
                            },
                            "files": result_files
                        }
                else:
                    return {
                        "job_id": job_id,
                        "status": "completed",
                        "message": "Processing complete but no SOAP note found",
                        "transcript": {
                            "raw": jobs[job_id].get("raw_transcript", "")
                        },
                        "files": result_files
                    }
            except Exception as e:
                logging.error(f"Error handling results: {str(e)}")
                return {
                    "job_id": job_id,
                    "status": "error",
                    "message": f"Error handling results: {str(e)}",
                    "transcript": {
                        "raw": jobs[job_id].get("raw_transcript", "")
                    }
                }
                
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            traceback.print_exc()
            return JSONResponse(status_code=500, content={
                "job_id": job_id,
                "status": "error",
                "message": str(e)
            })
        finally:
            # Force garbage collection
            gc.collect()
            log_memory_usage(f"End of job {job_id}")
    else:
        # Use background task if not waiting
        background_tasks = BackgroundTasks()
        background_tasks.add_task(process_audio_to_soap_task, job_id, audio_path, result_dir)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Audio processing started"
        }

# ADD THE AUDIO PROCESSING TASK
async def process_audio_to_soap_task(job_id, audio_path, result_dir):
    """Process audio file to SOAP note with less memory usage"""
    try:
        # Update job status
        jobs[job_id]["status"] = "transcribing"
        jobs[job_id]["message"] = "Transcribing audio..."
        
        # Force garbage collection before transcription
        gc.collect()
        
        # Step 1: Transcribe audio with Whisper directly to text
        transcript_text = transcribe_with_whisper(audio_path)
        if isinstance(transcript_text, dict) and "text" in transcript_text:
            transcript_text = transcript_text["text"]
        
        # Save transcript
        transcript_path = os.path.join(result_dir, "transcript.txt")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(str(transcript_text))
        
        # Store transcript content for response
        jobs[job_id]["raw_transcript"] = str(transcript_text)
        
        # Update job status
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Processing transcript..."
        
        # Force garbage collection before NLP processing
        gc.collect()
        
        # Process with simplified pipeline
        try:
            # Create a temporary file with just the text
            temp_transcript_path = os.path.join(result_dir, "transcript_clean.txt")
            with open(temp_transcript_path, 'w', encoding='utf-8') as f:
                f.write(str(transcript_text))
            
            # Process the transcript
            result = pipeline.process_transcript(temp_transcript_path, result_dir)
            
            # Update job status with results
            if result:
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["completed_at"] = datetime.now().isoformat()
                
                # Collect result file paths
                result_files = {}
                if "notes" in result:
                    for format_name, file_path in result["notes"].items():
                        if file_path and os.path.exists(file_path):
                            result_files[format_name] = file_path
                
                jobs[job_id]["result_files"] = result_files
                jobs[job_id]["message"] = "Successfully generated SOAP note"
            else:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["message"] = "Failed to process transcript"
                jobs[job_id]["completed_at"] = datetime.now().isoformat()
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = f"Error in NLP processing: {str(e)}"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        import traceback
        traceback.print_exc()


@app.post("/transcript/process")
async def process_transcript(
    background_tasks: BackgroundTasks,
    transcript: TranscriptRequest,
    wait_for_result: bool = False,
    timeout_seconds: int = 300  # 5 minutes default timeout
):
    """Process a text transcript and generate clinical notes"""
    job_id = generate_job_id()
    job_name = transcript.job_name or f"Transcript Job {job_id[:8]}"
    
    # Save transcript to file
    transcript_path = save_transcript(transcript.text, job_id)
    
    # Create output directory for this job
    job_output_dir = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)
    
    # Create job entry
    jobs[job_id] = {
        "job_id": job_id,
        "name": job_name,
        "status": "processing",
        "created_at": datetime.now().isoformat(),
        "transcript_path": transcript_path
    }
    
    # Process directly instead of using background tasks if waiting for result
    if wait_for_result:
        try:
            # Process the transcript
            result = pipeline.process_transcript(transcript_path, job_output_dir)
            
            # Update job status with results
            if result:
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["completed_at"] = datetime.now().isoformat()
                
                # Collect result file paths
                result_files = {}
                if "notes" in result:
                    for format_name, file_path in result["notes"].items():
                        if file_path and os.path.exists(file_path):
                            result_files[format_name] = file_path
                
                jobs[job_id]["result_files"] = result_files
                
                # Return SOAP note JSON
                soap_path = result_files.get("json")
                if soap_path and os.path.exists(soap_path):
                    with open(soap_path, 'r', encoding='utf-8') as f:
                        soap_data = json.load(f)
                    return {
                        "job_id": job_id,
                        "status": "completed",
                        "soap_note": soap_data,
                        "completed_at": jobs[job_id]["completed_at"],
                        "files": result_files
                    }
                else:
                    return {
                        "job_id": job_id,
                        "status": "completed",
                        "message": "Process completed but SOAP note file not found",
                        "files": result_files
                    }
            else:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["message"] = "Failed to process transcript"
                jobs[job_id]["completed_at"] = datetime.now().isoformat()
                
                return JSONResponse(status_code=500, content={
                    "job_id": job_id,
                    "status": "failed",
                    "message": "Failed to process transcript"
                })
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = str(e)
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
            return JSONResponse(status_code=500, content={
                "job_id": job_id,
                "status": "failed",
                "message": str(e)
            })
    else:
        # Start background processing
        background_tasks.add_task(process_transcript_task, job_id, transcript_path)
        
        return JSONResponse(status_code=202, content={
            "job_id": job_id,
            "status": "queued",
            "message": "Transcript processing started",
            "created_at": jobs[job_id]["created_at"]
        })

@app.post("/transcript/upload", response_model=JobStatus)
async def upload_transcript(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_name: str = Form(None)
):
    """Upload and process a transcript file"""
    job_id = generate_job_id()
    job_name = job_name or f"Upload Job {job_id[:8]}"
    
    # Save uploaded file
    file_path = save_uploaded_file(file, job_id)
    
    # Create job entry
    jobs[job_id] = {
        "job_id": job_id,
        "name": job_name,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "transcript_path": file_path
    }
    
    # Start background processing
    background_tasks.add_task(process_transcript_task, job_id, file_path)
    
    return JSONResponse(status_code=202, content={
        "job_id": job_id,
        "status": "queued",
        "message": "File uploaded and processing started",
        "created_at": jobs[job_id]["created_at"]
    })

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id].copy()
    
    # Include only the necessary fields for the response
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "message": job.get("message"),
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at"),
        "result_files": job.get("result_files")
    }

@app.get("/jobs")
async def list_jobs():
    """List all processing jobs"""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "name": job_info.get("name", "Unnamed Job"),
                "status": job_info["status"],
                "created_at": job_info["created_at"],
                "completed_at": job_info.get("completed_at")
            }
            for job_id, job_info in jobs.items()
        ]
    }

@app.get("/results/{job_id}/{file_type}")
async def get_result_file(job_id: str, file_type: str):
    """Get a specific result file for a job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if "result_files" not in job or file_type not in job["result_files"]:
        raise HTTPException(status_code=404, detail=f"Result file '{file_type}' not found")
    
    file_path = job["result_files"][file_type]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")
    
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type="application/octet-stream"
    )

@app.post("/prescription/suggest", response_model=PrescriptionResponse)
async def suggest_prescription(
    request: PrescriptionRequest,
    prescription_api_url: str = "http://localhost:8000"
):
    """Get prescription suggestions based on patient data"""
    patient_data = request.patient_data.dict()
    
    # Get prescription from external API
    prescription = get_prescription(patient_data, prescription_api_url)
    
    if not prescription:
        raise HTTPException(status_code=500, detail="Failed to get prescription suggestions")
    
    return prescription

@app.post("/soap-to-prescription/{job_id}", response_model=PrescriptionResponse)
async def soap_to_prescription(
    job_id: str,
    prescription_api_url: str = "http://localhost:8000"
):
    """Get prescription suggestions from a job's SOAP note"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if "result_files" not in job or "json" not in job["result_files"]:
        raise HTTPException(status_code=404, detail="SOAP note JSON not found")
    
    soap_note_path = job["result_files"]["json"]
    
    # Extract patient data from SOAP note
    patient_data = extract_patient_data_from_soap(soap_note_path)
    
    if not patient_data:
        raise HTTPException(status_code=500, detail="Failed to extract patient data from SOAP note")
    
    # Get prescription from external API
    prescription = get_prescription(patient_data, prescription_api_url)
    
    if not prescription:
        raise HTTPException(status_code=500, detail="Failed to get prescription suggestions")
    
    # Save prescription to result directory
    prescription_path = os.path.join(RESULTS_DIR, job_id, "prescription.json")
    with open(prescription_path, "w", encoding="utf-8") as f:
        json.dump(prescription, f, indent=2)
    
    # Add prescription file to job results
    jobs[job_id]["result_files"]["prescription"] = prescription_path
    
    return prescription

@app.post("/complete-pipeline", response_model=Dict[str, Any])
async def run_complete_pipeline(
    background_tasks: BackgroundTasks,
    transcript: TranscriptRequest,
    prescription_api_url: str = "http://localhost:8000"
):
    """Run complete pipeline from transcript to prescription"""
    # First process the transcript
    job_id = generate_job_id()
    job_name = transcript.job_name or f"Complete Pipeline Job {job_id[:8]}"
    
    # Save transcript to file
    transcript_path = save_transcript(transcript.text, job_id)
    
    # Create job entry
    jobs[job_id] = {
        "job_id": job_id,
        "name": job_name,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "transcript_path": transcript_path,
        "pipeline_type": "complete"
    }
    
    # Define the complete pipeline task
    async def complete_pipeline_task(job_id, transcript_path):
        try:
            # Update job status
            jobs[job_id]["status"] = "processing_transcript"
            
            # Create output directory for this job
            job_output_dir = os.path.join(RESULTS_DIR, job_id)
            os.makedirs(job_output_dir, exist_ok=True)
            
            # Process the transcript
            result = pipeline.process_transcript(transcript_path, job_output_dir)
            
            if not result or "notes" not in result:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["message"] = "Failed to generate clinical notes"
                return
            
            # Collect result file paths
            result_files = {}
            for format_name, file_path in result["notes"].items():
                if file_path and os.path.exists(file_path):
                    result_files[format_name] = file_path
            
            jobs[job_id]["result_files"] = result_files
            
            # Update status to processing prescription
            jobs[job_id]["status"] = "processing_prescription"
            
            # Get SOAP note path
            soap_note_path = result["notes"].get("json")
            if not soap_note_path or not os.path.exists(soap_note_path):
                jobs[job_id]["status"] = "partial_success"
                jobs[job_id]["message"] = "Generated clinical notes but SOAP note JSON not found for prescription"
                jobs[job_id]["completed_at"] = datetime.now().isoformat()
                return
            
            # Extract patient data from SOAP note
            patient_data = extract_patient_data_from_soap(soap_note_path)
            
            if not patient_data:
                jobs[job_id]["status"] = "partial_success"
                jobs[job_id]["message"] = "Generated clinical notes but failed to extract patient data"
                jobs[job_id]["completed_at"] = datetime.now().isoformat()
                return
            
            # Get prescription
            prescription = get_prescription(patient_data, prescription_api_url)
            
            if not prescription:
                jobs[job_id]["status"] = "partial_success"
                jobs[job_id]["message"] = "Generated clinical notes but failed to get prescription"
                jobs[job_id]["completed_at"] = datetime.now().isoformat()
                return
            
            # Save prescription to result directory
            prescription_path = os.path.join(job_output_dir, "prescription.json")
            with open(prescription_path, "w", encoding="utf-8") as f:
                json.dump(prescription, f, indent=2)
            
            # Add prescription file to job results
            jobs[job_id]["result_files"]["prescription"] = prescription_path
            
            # Update job status
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["message"] = "Successfully completed full pipeline"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = str(e)
            import traceback
            traceback.print_exc()
    
    # Start the task
    background_tasks.add_task(complete_pipeline_task, job_id, transcript_path)
    
    return JSONResponse(status_code=202, content={
        "job_id": job_id,
        "status": "queued",
        "message": "Complete pipeline processing started",
        "created_at": jobs[job_id]["created_at"]
    })

# Run the API with uvicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)