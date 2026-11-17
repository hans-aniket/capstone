"""
backend/api/router.py
───────────────────────
FastAPI router containing endpoints for:
1. Comparative NLP inference across all model tiers
2. Model explainability (XAI) & token attribution heatmaps
3. Aspect-Based Sentiment Analysis (ABSA)
4. Robustness, Sarcasm & Adversarial Stress-Testing
5. Batch processing & business intelligence analytics
"""

import io
import csv
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from fastapi.responses import StreamingResponse

from backend.schemas.inference import (
    InferenceRequest,
    ComparativeInferenceResponse,
    SystemHealthResponse,
    ExplainResponse,
    ABSAResponse,
    StressTestRequest,
    StressTestResponse,
    RobustnessBenchmarkResponse,
    BatchAnalysisResponse
)
from backend.services.inference_service import inference_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["sentiment-analysis"])


# ── System Health ─────────────────────────────────────────────────────────────

@router.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """
    Check the health of the API and list all currently loaded models in memory.
    """
    models_info = inference_service.get_loaded_models_info()
    return SystemHealthResponse(
        status="online" if models_info else "starting",
        loaded_models=models_info
    )


# ── 1. Comparative Inference (Enhanced with ABSA & XAI) ───────────────────────

@router.post("/predict", response_model=ComparativeInferenceResponse)
async def predict_sentiment(request: InferenceRequest):
    """
    Run comparative sentiment analysis across all model tiers concurrently,
    enhanced with token attribution explainability and aspect-based breakdown.
    """
    text = request.text.strip() if request.text else ""
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text cannot be empty."
        )

    try:
        predictions, errors, total_latency = inference_service.run_comparative_inference(text)
        
        # Attach ABSA
        aspect_res = None
        try:
            aspect_res = inference_service.get_aspect_analysis(text)
        except Exception as e:
            logger.warning("ABSA extraction skipped: %s", e)

        # Attach Explainability
        explain_res = None
        try:
            explain_res = inference_service.get_explanation(text, request.model)
        except Exception as e:
            logger.warning("Explainability skipped: %s", e)

        return ComparativeInferenceResponse(
            text=text,
            predictions=predictions,
            total_latency_ms=total_latency,
            errors=errors,
            aspect_analysis=aspect_res,
            explanation=explain_res
        )
    except Exception as e:
        logger.error("Inference pipeline failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failed: {str(e)}"
        )


# ── 2. Explainability (XAI) ───────────────────────────────────────────────────

@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(request: InferenceRequest):
    """
    Generates token-level attribution weights for inline visual heatmaps.
    Supports DistilBERT gradient saliency on GPU and Classical TF-IDF feature attribution.
    """
    text = request.text.strip() if request.text else ""
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text cannot be empty."
        )

    try:
        return inference_service.get_explanation(text, request.model)
    except Exception as e:
        logger.error("Explainability failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability generation failed: {str(e)}"
        )


# ── 3. Aspect-Based Sentiment Analysis (ABSA) ─────────────────────────────────

@router.post("/absa", response_model=ABSAResponse)
async def extract_aspects(request: InferenceRequest):
    """
    Extracts fine-grained sentiment on product dimensions
    (Battery, Build Quality, Performance, Price, Design, Support, Sound/Display).
    """
    text = request.text.strip() if request.text else ""
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text cannot be empty."
        )

    try:
        return inference_service.get_aspect_analysis(text)
    except Exception as e:
        logger.error("ABSA failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Aspect analysis failed: {str(e)}"
        )


# ── 4. Robustness & Adversarial Stress-Testing ────────────────────────────────

@router.get("/robustness/benchmark", response_model=RobustnessBenchmarkResponse)
async def get_robustness_benchmark(refresh: bool = False):
    """
    Runs or returns the cached CheckList benchmark suite evaluating model resilience
    against Negation, Sarcasm, Contrastive Conjunctions, Typos, and Length Extremes.
    """
    try:
        return inference_service.get_robustness_benchmark(force_refresh=refresh)
    except Exception as e:
        logger.error("Benchmark failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Robustness benchmark failed: {str(e)}"
        )


@router.post("/robustness/stress-test", response_model=StressTestResponse)
async def stress_test_text(request: StressTestRequest):
    """
    Dynamically generates linguistic perturbations (typos, sarcasm, negation flips)
    on the user's input and tests whether models remain invariant or flip predictions.
    """
    text = request.text.strip() if request.text else ""
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty."
        )

    try:
        return inference_service.run_stress_test(text)
    except Exception as e:
        logger.error("Stress-test failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dynamic stress-test failed: {str(e)}"
        )


# ── 5. Batch Processing & Business Intelligence ───────────────────────────────

@router.post("/batch/upload", response_model=BatchAnalysisResponse)
async def upload_batch_csv(
    file: UploadFile = File(...),
    primary_model: Optional[str] = Form("DistilBERT")
):
    """
    Upload a CSV of customer reviews to perform batch sentiment analysis,
    Net Promoter Score (NPS) estimation, aspect breakdown, and keyword driver extraction.
    """
    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are currently supported for batch analytics."
        )

    try:
        content = await file.read()
        return inference_service.process_batch_csv(content, primary_model=primary_model or "DistilBERT")
    except Exception as e:
        logger.error("Batch processing failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.get("/batch/sample", response_model=BatchAnalysisResponse)
async def get_sample_batch_analytics():
    """
    Instantly returns pre-processed batch analytics for the authentic Amazon reviews demo dataset.
    """
    try:
        return inference_service.get_sample_batch()
    except Exception as e:
        logger.error("Sample batch failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load sample batch: {str(e)}"
        )


@router.post("/batch/export")
async def export_batch_csv(data: BatchAnalysisResponse):
    """
    Generates and streams a downloadable analyzed CSV file with predicted sentiments,
    confidence ratings, and primary aspect tags.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Product", "Review Text", "Predicted Sentiment", "Confidence (%)", "Primary Aspect"])
    
    for row in data.reviews:
        writer.writerow([
            row.id,
            row.product or "N/A",
            row.text,
            row.sentiment,
            row.confidence,
            row.primary_aspect
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sentiment_batch_analysis.csv"}
    )
