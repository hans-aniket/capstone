"""
backend/services/inference_service.py
──────────────────────────────────────
Service layer wrapping the UnifiedPredictor to maintain state,
provide efficient in-memory inference, and safely handle errors.
"""

import time
import logging
from typing import Dict, Any, List

from evaluate.unified_inference import UnifiedPredictor

logger = logging.getLogger(__name__)

class InferenceService:
    """
    Singleton-style service wrapping the UnifiedPredictor for the FastAPI backend.
    Ensures models are loaded once and shared across requests.
    """
    def __init__(self):
        self._predictor = None
        self._is_loaded = False

    def load_models(self):
        """Initializes all models into memory. Call this during app startup."""
        if self._is_loaded:
            return
        
        logger.info("[InferenceService] Initializing UnifiedPredictor...")
        self._predictor = UnifiedPredictor()
        self._is_loaded = True
        logger.info("[InferenceService] All models successfully loaded into memory.")

    def get_loaded_models_info(self) -> List[Dict[str, Any]]:
        """Returns metadata about the currently loaded models."""
        if not self._is_loaded or not self._predictor:
            return []

        info = []
        for name, predictor in self._predictor.predictors.items():
            info.append({
                "name": name,
                "tier": getattr(predictor, "tier", "unknown"),
                "status": "ready",
                "vocab_size": getattr(getattr(predictor, "vocabulary", None), "vocab_size", None) or 
                              getattr(getattr(predictor, "preprocessor", None), "vocab_size", None),
                "device": str(getattr(predictor, "device", "cpu"))
            })
        return info

    def run_comparative_inference(self, text: str) -> tuple[List[Dict[str, Any]], List[str], float]:
        """
        Runs the text through all models.
        Returns (predictions, errors, total_latency_ms).
        """
        if not self._is_loaded or not self._predictor:
            raise RuntimeError("Models are not loaded. Service is unavailable.")

        t0 = time.perf_counter()
        predictions = []
        errors = []

        # Run inference across all predictors
        for name, predictor in self._predictor.predictors.items():
            try:
                res = predictor.predict_single(text)
                predictions.append({
                    "model": name,
                    "tier": getattr(predictor, "tier", "unknown"),
                    "label": res["label"],
                    "label_name": res["label_name"],
                    "confidence": res["confidence"],
                    "latency_ms": res["latency_ms"]
                })
            except Exception as e:
                logger.error("Failed inference for %s: %s", name, e)
                errors.append(f"{name}: {str(e)}")

        total_latency = (time.perf_counter() - t0) * 1000
        
        return predictions, errors, total_latency

# Instantiate a global singleton service for the API
inference_service = InferenceService()
