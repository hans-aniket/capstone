"""
backend/schemas/inference.py
──────────────────────────────
Pydantic schemas for the comparative inference API.
"""

from typing import Dict, List, Any
from pydantic import BaseModel, Field

class InferenceRequest(BaseModel):
    text: str = Field(..., description="The product review text to analyze")

class ModelPrediction(BaseModel):
    model: str = Field(..., description="Name of the model")
    tier: str = Field(..., description="Architecture tier (classical, deep_learning, transformers)")
    label: int = Field(..., description="Integer label (0 for negative, 1 for positive)")
    label_name: str = Field(..., description="Human-readable label name")
    confidence: float = Field(..., description="Confidence score (probability of positive class, 0.0 to 1.0)")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")

class ComparativeInferenceResponse(BaseModel):
    text: str = Field(..., description="The original input text")
    predictions: List[ModelPrediction] = Field(..., description="List of predictions from all available models")
    total_latency_ms: float = Field(..., description="Total time taken to run all models")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered during inference")

class ModelHealthInfo(BaseModel):
    name: str
    tier: str
    status: str
    vocab_size: int | None = None
    device: str | None = None

class SystemHealthResponse(BaseModel):
    status: str
    loaded_models: List[ModelHealthInfo]
