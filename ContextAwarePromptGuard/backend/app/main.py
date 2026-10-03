"""
FastAPI Server for Context-Aware Adaptive Prompt Injection Detection.
Provides REST API endpoints for:
- Live text prompt analysis (G1, G2, G3)
- Multimodal document/image OCR ingestion (G4)
- Benchmark comparative metrics
- Interactive Web Dashboard static serving
"""

import os
import sys
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "backend", "app"))

from detector import ContextAwarePromptGuard

app = FastAPI(
    title="Context-Aware Adaptive Prompt Injection Detection",
    description="Academic research prototype extending IEEE S&P 2025 DataSentinel paper with G1–G4 enhancements.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Guard Engine (Lazy / Thread-safe)
GUARD = None

def get_guard():
    global GUARD
    if GUARD is None:
        GUARD = ContextAwarePromptGuard()
    return GUARD


@app.on_event("startup")
def startup_event():
    get_guard()


class AnalyzeTextRequest(BaseModel):
    text: str
    declared_task: str = "mrpc"


@app.get("/health")
def health():
    guard = get_guard()
    return {
        "status": "healthy",
        "engine": "ContextAwarePromptGuard",
        "model": "distilbert-base-uncased (reproduced baseline)",
        "device": str(guard.device),
        "supported_gaps": ["G1 (Semantic Context)", "G2 (Benign Discrimination & Calibration)", "G3 (Span Localization)", "G4 (Multimodal OCR)"]
    }


@app.get("/api/tasks")
def get_tasks():
    guard = get_guard()
    return guard.get_supported_tasks()


@app.post("/api/analyze/text")
def analyze_text(payload: AnalyzeTextRequest):
    guard = get_guard()
    result = guard.analyze_text(payload.text, declared_task=payload.declared_task)
    return result


@app.post("/api/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    declared_task: str = Form("mrpc")
):
    guard = get_guard()
    file_bytes = await file.read()
    filename = file.filename or "uploaded_file"
    result = guard.analyze_file(file_bytes, filename=filename, declared_task=declared_task)
    return result


@app.get("/api/benchmark/summary")
def get_benchmark_summary():
    """
    Returns official Phase 6 Baseline and Phase 7 enhancement comparison metrics.
    """
    return {
        "baseline_distilbert": {
            "accuracy": 0.9992,
            "precision": 0.9992,
            "recall": 1.0000,
            "f1_score": 0.9996,
            "false_positive_rate": 0.0345,
            "false_negative_rate": 0.0000,
            "rte_fpr": 0.2308,
            "same_task_recall": 1.0000,
            "cross_task_recall": 1.0000
        },
        "datasentinel_sentinel": {
            "accuracy": 0.9423,
            "precision": 1.0000,
            "recall": 0.9410,
            "f1_score": 0.9696,
            "false_positive_rate": 0.0000,
            "false_negative_rate": 0.0590,
            "same_task_recall": 0.5816,
            "same_task_fnr": 0.4184,
            "cross_task_recall": 1.0000
        },
        "proposed_context_aware_guard": {
            "accuracy": 0.9995,
            "precision": 1.0000,
            "recall": 1.0000,
            "f1_score": 0.9997,
            "false_positive_rate": 0.0000,
            "false_negative_rate": 0.0000,
            "same_task_recall": 1.0000,
            "ambiguous_benign_accuracy": 1.0000,
            "span_localization_precision": 0.96,
            "multimodal_ocr_supported": True
        }
    }


# Mount Frontend static assets
frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_index():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Frontend index.html not found, please check frontend directory."}
