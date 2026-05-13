"""
backend/api/router.py
───────────────────────
FastAPI router containing the endpoints for comparative NLP inference.
"""

from fastapi import APIRouter, HTTPException, status
from backend.schemas.inference import InferenceRequest, ComparativeInferenceResponse, SystemHealthResponse
from backend.services.inference_service import inference_service

router = APIRouter(prefix="/api/v1", tags=["inference"])

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

@router.post("/predict", response_model=ComparativeInferenceResponse)
async def predict_sentiment(request: InferenceRequest):
    """
    Run comparative sentiment analysis on the provided product review text.
    Executes inference across all loaded models (Classical, DL, Transformers) concurrently.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text cannot be empty."
        )

    try:
        predictions, errors, total_latency = inference_service.run_comparative_inference(request.text)
        
        return ComparativeInferenceResponse(
            text=request.text,
            predictions=predictions,
            total_latency_ms=total_latency,
            errors=errors
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failed: {str(e)}"
        )
