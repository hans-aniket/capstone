"""
backend/schemas/inference.py
──────────────────────────────
Pydantic schemas for the comparative inference and advanced analytics API.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    text: str = Field(..., description="The product review text to analyze")
    model: Optional[str] = Field(None, description="Optional target model name for detailed explainability")


class ModelPrediction(BaseModel):
    model: str = Field(..., description="Name of the model")
    tier: str = Field(..., description="Architecture tier (classical, deep_learning, transformers)")
    label: int = Field(..., description="Integer label (0 for negative, 1 for positive)")
    label_name: str = Field(..., description="Human-readable label name")
    confidence: float = Field(..., description="Confidence score (probability of positive class, 0.0 to 1.0)")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class TokenAttribution(BaseModel):
    word: str
    score: float
    sentiment: str  # positive, negative, neutral
    importance: float


class ExplainResponse(BaseModel):
    model: str
    tier: str
    predicted_label: str
    confidence: float
    tokens: List[TokenAttribution]
    top_positive: List[str]
    top_negative: List[str]


class AspectOpinion(BaseModel):
    aspect: str
    keywords_matched: List[str]
    sentiment: str
    confidence: float
    polarity_score: float
    snippet: str


class ABSAResponse(BaseModel):
    aspects_found: int
    dominant_tone: str
    aspects: List[AspectOpinion]


class ComparativeInferenceResponse(BaseModel):
    text: str = Field(..., description="The original input text")
    predictions: List[ModelPrediction] = Field(..., description="List of predictions from all available models")
    total_latency_ms: float = Field(..., description="Total time taken to run all models")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered during inference")
    aspect_analysis: Optional[ABSAResponse] = None
    explanation: Optional[ExplainResponse] = None


class ModelHealthInfo(BaseModel):
    name: str
    tier: str
    status: str
    vocab_size: int | None = None
    device: str | None = None


class SystemHealthResponse(BaseModel):
    status: str
    loaded_models: List[ModelHealthInfo]


# ── Robustness & Stress-Testing Schemas ─────────────────────────────────────────

class PerturbationResult(BaseModel):
    perturbation_type: str
    description: str
    text: str
    predictions: Dict[str, Any]


class StressTestRequest(BaseModel):
    text: str


class StressTestResponse(BaseModel):
    original_text: str
    perturbation_count: int
    results: List[PerturbationResult]


class RobustnessBenchmarkResponse(BaseModel):
    total_test_cases: int
    categories: List[str]
    leaderboard: List[Dict[str, Any]]
    category_breakdown: List[Dict[str, Any]]
    cases: List[Dict[str, Any]]


# ── Batch Analytics Schemas ───────────────────────────────────────────────────

class BatchReviewItem(BaseModel):
    id: int
    product: Optional[str] = None
    text: str
    sentiment: str
    confidence: float
    primary_aspect: str
    aspects: List[Dict[str, Any]]


class NPSMetrics(BaseModel):
    nps_score: float
    promoters_pct: float
    passives_pct: float
    detractors_pct: float
    classification: str


class BatchAnalysisResponse(BaseModel):
    total_reviews: int
    positive_count: int
    negative_count: int
    positive_pct: float
    negative_pct: float
    nps_metrics: NPSMetrics
    top_positive_drivers: List[Dict[str, Any]]
    top_complaint_themes: List[Dict[str, Any]]
    aspect_breakdown: List[Dict[str, Any]]
    reviews: List[BatchReviewItem]
