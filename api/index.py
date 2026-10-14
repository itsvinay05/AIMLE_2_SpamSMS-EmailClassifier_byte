"""
Vercel Serverless API - Spam SMS / Email Classifier
---------------------------------------------------
FastAPI application compatible with Vercel Serverless Functions.
Provides /api/predict, /api/batch_predict, /api/parse_email, and /api/metrics.
"""

import os
import json
import re
import math
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Spam SMS & Email Classifier API (Vercel)",
    description="High-performance NLP microservice for SMS/Email Spam vs Ham detection.",
    version="1.0.0"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache loaded model data in memory across warm serverless invocations
MODEL_CACHE = {}


def load_model_data():
    """Loads model configuration and precomputed parameters."""
    if "model" in MODEL_CACHE:
        return MODEL_CACHE["model"]

    # Search possible paths for model_export.json
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "public", "model_export.json"),
        os.path.join(os.path.dirname(__file__), "model_export.json"),
        os.path.join("public", "model_export.json")
    ]

    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_compiled_regex"] = re.compile(data["token_pattern"], re.UNICODE)
                MODEL_CACHE["model"] = data
                return data

    raise RuntimeError("Model export file (model_export.json) not found.")


def classify_text(text: str, model_data: dict, threshold: float = 0.5, explain: bool = False):
    """
    Computes exact TF-IDF + Multinomial Naive Bayes posterior probabilities,
    decision classification with custom threshold, and word feature contributions.
    """
    vocab = model_data["vocabulary"]
    priors = model_data["class_log_prior"]
    token_pattern = model_data["_compiled_regex"]

    tokens = token_pattern.findall(text.lower())
    all_terms = list(tokens)
    for i in range(len(tokens) - 1):
        all_terms.append(f"{tokens[i]} {tokens[i+1]}")

    tf = {}
    for t in all_terms:
        if t in vocab:
            tf[t] = tf.get(t, 0) + 1

    # L2 normalization of TF * IDF
    norm_sq = 0.0
    for t, count in tf.items():
        idf = vocab[t][0]
        val = count * idf
        norm_sq += val * val

    norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0

    jll_ham = priors[0]
    jll_spam = priors[1]
    contributions = []

    for t, count in tf.items():
        idf, diff, ham_lp, spam_lp = vocab[t]
        weight = (count * idf) / norm
        jll_ham += weight * ham_lp
        jll_spam += weight * spam_lp

        if explain:
            contributions.append({
                "term": t,
                "weight": round(weight, 4),
                "spam_signal": round(diff * weight, 4),
                "favors": "Spam" if diff > 0 else "Ham"
            })

    # LogSumExp for probabilities
    max_jll = max(jll_ham, jll_spam)
    sum_exp = math.exp(jll_ham - max_jll) + math.exp(jll_spam - max_jll)
    log_sum = max_jll + math.log(sum_exp)

    prob_ham = math.exp(jll_ham - log_sum)
    prob_spam = math.exp(jll_spam - log_sum)

    # Threshold-based decision
    predicted_label = "Spam" if prob_spam >= threshold else "Ham"
    confidence = prob_spam if predicted_label == "Spam" else prob_ham

    if explain:
        contributions.sort(key=lambda x: abs(x["spam_signal"]), reverse=True)

    return {
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4),
        "ham_probability": round(prob_ham, 4),
        "spam_probability": round(prob_spam, 4),
        "contributions": contributions[:10] if explain else None
    }


# Schemas
class MessageRequest(BaseModel):
    text: str = Field(..., description="Message text or email content to classify.")
    threshold: Optional[float] = Field(0.50, ge=0.01, le=0.99, description="Spam decision threshold.")
    explain: bool = Field(False, description="Return top feature contributions.")


class BatchRequest(BaseModel):
    messages: List[str] = Field(..., description="List of messages to classify.")
    threshold: Optional[float] = Field(0.50, ge=0.01, le=0.99)


@app.get("/api")
def api_root():
    return {
        "service": "Spam SMS / Email Classifier (Vercel Edition)",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "predict": "/api/predict",
            "batch_predict": "/api/batch_predict",
            "metrics": "/api/metrics",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
def health():
    try:
        model = load_model_data()
        return {
            "status": "healthy",
            "model_loaded": True,
            "vocab_size": model["vocab_size"]
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }


@app.get("/api/metrics")
def get_metrics():
    model = load_model_data()
    return {
        "model_name": model.get("model_name"),
        "metrics": model.get("metrics"),
        "confusion_matrix": model.get("confusion_matrix"),
        "test_distribution": model.get("test_distribution")
    }


@app.post("/api/predict")
def predict(req: MessageRequest):
    start = time.perf_counter()
    model = load_model_data()
    res = classify_text(req.text, model, threshold=req.threshold, explain=req.explain)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "input_text": req.text,
        "predicted_label": res["predicted_label"],
        "confidence": res["confidence"],
        "ham_probability": res["ham_probability"],
        "spam_probability": res["spam_probability"],
        "threshold": req.threshold,
        "latency_ms": latency_ms,
        "explanation": res["contributions"]
    }


@app.post("/api/batch_predict")
def batch_predict(req: BatchRequest):
    start = time.perf_counter()
    model = load_model_data()
    results = []
    spam_count = 0
    ham_count = 0

    for msg in req.messages:
        res = classify_text(msg, model, threshold=req.threshold, explain=False)
        if res["predicted_label"] == "Spam":
            spam_count += 1
        else:
            ham_count += 1
        results.append({
            "text": msg,
            "predicted_label": res["predicted_label"],
            "confidence": res["confidence"],
            "spam_probability": res["spam_probability"]
        })

    total_latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "total_messages": len(req.messages),
        "spam_count": spam_count,
        "ham_count": ham_count,
        "total_latency_ms": total_latency_ms,
        "results": results
    }


# Static file serving for standalone / local preview
public_dir = os.path.join(os.path.dirname(__file__), "..", "public")
if os.path.exists(public_dir):
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    @app.get("/")
    def index():
        return FileResponse(os.path.join(public_dir, "index.html"))

    app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")

