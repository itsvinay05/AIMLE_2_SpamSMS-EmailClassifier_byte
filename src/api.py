"""
Spam SMS / Email Classifier - Production REST API Service
---------------------------------------------------------
FastAPI microservice exposing low-latency inference endpoints for single
and batch spam classification, email parsing, and health metrics.
"""

import json
import os
import time
from typing import List, Optional, Dict, Any
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.predict import predict_message, load_model
from src.email_parser import parse_email_message

# Initialize FastAPI App
app = FastAPI(
    title="Spam SMS & Email Classification API",
    description="High-performance, explainable NLP text classification microservice for Spam vs Ham detection.",
    version="1.0.0"
)

# Global model store
MODELS: Dict[str, Any] = {}
DEFAULT_MODEL_PATH = os.path.join("models", "spam_classifier.joblib")


@app.on_event("startup")
def startup_event():
    """Pre-load model pipeline into memory on server boot."""
    if os.path.exists(DEFAULT_MODEL_PATH):
        MODELS["default"] = load_model(DEFAULT_MODEL_PATH)
    else:
        print(f"Warning: Model not found at {DEFAULT_MODEL_PATH}. Run training first.")


# Pydantic Schemas
class MessageRequest(BaseModel):
    text: str = Field(..., description="Message text or raw email content to classify.", example="Congratulations! You have won a prize. Claim now.")
    is_email: bool = Field(False, description="Set to true if input contains email headers / HTML body.")
    explain: bool = Field(False, description="Set to true to return word-level feature contributions.")
    threshold: Optional[float] = Field(0.50, ge=0.01, le=0.99, description="Custom spam decision threshold (default: 0.50).")


class BatchMessageRequest(BaseModel):
    messages: List[str] = Field(..., description="List of messages to classify.", example=["Hello there", "URGENT! Claim prize"])
    threshold: Optional[float] = Field(0.50, ge=0.01, le=0.99)


class PredictionResponse(BaseModel):
    input_text: str
    predicted_label: str
    confidence: float
    ham_probability: float
    spam_probability: float
    latency_ms: float
    is_email: bool
    email_metadata: Optional[Dict[str, Any]] = None
    explanation: Optional[List[Dict[str, Any]]] = None


class BatchPredictionResponse(BaseModel):
    total_messages: int
    spam_count: int
    ham_count: int
    total_latency_ms: float
    results: List[Dict[str, Any]]


@app.get("/")
def root():
    """API overview and links."""
    return {
        "service": "Spam SMS & Email Classification API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/batch_predict",
            "metrics": "/api/v1/metrics",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    """Health check probe."""
    return {
        "status": "healthy",
        "model_loaded": "default" in MODELS
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_single(req: MessageRequest):
    """Classifies a single SMS or email message."""
    if "default" not in MODELS:
        try:
            MODELS["default"] = load_model(DEFAULT_MODEL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model unavailable: {e}")

    start_time = time.perf_counter()
    model = MODELS["default"]
    
    text_to_process = req.text
    email_meta = None

    if req.is_email:
        parsed = parse_email_message(req.text)
        text_to_process = parsed["combined_text"]
        email_meta = {
            "subject": parsed["subject"],
            "from": parsed["from"],
            "has_ip_url": parsed["has_ip_url"],
            "extracted_urls": parsed["extracted_urls"]
        }

    raw_result = predict_message(model, text_to_process, explain=req.explain)
    
    # Custom decision threshold
    spam_prob = raw_result["spam_probability"]
    ham_prob = raw_result["ham_probability"]
    is_spam = spam_prob >= req.threshold
    label_str = "Spam" if is_spam else "Ham"
    confidence = spam_prob if is_spam else ham_prob

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    explanation_list = None
    if req.explain and "explanation" in raw_result:
        explanation_list = [{"term": t, "impact_score": round(float(w), 4)} for t, w in raw_result["explanation"]]

    return PredictionResponse(
        input_text=req.text,
        predicted_label=label_str,
        confidence=round(confidence, 4),
        ham_probability=round(ham_prob, 4),
        spam_probability=round(spam_prob, 4),
        latency_ms=round(latency_ms, 2),
        is_email=req.is_email,
        email_metadata=email_meta,
        explanation=explanation_list
    )


@app.post("/api/v1/batch_predict", response_model=BatchPredictionResponse)
def predict_batch(req: BatchMessageRequest):
    """Classifies a batch of messages in parallel."""
    if "default" not in MODELS:
        try:
            MODELS["default"] = load_model(DEFAULT_MODEL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model unavailable: {e}")

    start_time = time.perf_counter()
    model = MODELS["default"]
    
    results = []
    spam_count = 0
    ham_count = 0

    classes = list(model.classes_)
    labels = model.predict(req.messages)
    probas = model.predict_proba(req.messages)

    for idx, msg in enumerate(req.messages):
        spam_prob = float(probas[idx][classes.index(1)])
        ham_prob = float(probas[idx][classes.index(0)])
        
        is_spam = spam_prob >= req.threshold
        label_str = "Spam" if is_spam else "Ham"
        confidence = spam_prob if is_spam else ham_prob

        if is_spam:
            spam_count += 1
        else:
            ham_count += 1

        results.append({
            "id": idx + 1,
            "message": msg,
            "predicted_label": label_str,
            "confidence": round(confidence, 4),
            "spam_probability": round(spam_prob, 4),
            "ham_probability": round(ham_prob, 4)
        })

    total_latency = (time.perf_counter() - start_time) * 1000.0

    return BatchPredictionResponse(
        total_messages=len(req.messages),
        spam_count=spam_count,
        ham_count=ham_count,
        total_latency_ms=round(total_latency, 2),
        results=results
    )


@app.get("/api/v1/metrics")
def get_model_metrics():
    """Returns saved model performance metrics on the test set."""
    metrics_path = os.path.join("outputs", "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics file not found. Run training pipeline first.")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)
