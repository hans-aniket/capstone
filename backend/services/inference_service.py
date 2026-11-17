"""
backend/services/inference_service.py
──────────────────────────────────────
Service layer wrapping the UnifiedPredictor to maintain state,
provide efficient in-memory inference, explainability, robustness,
aspect-based sentiment analysis, and batch analytics.
"""

import time
import logging
from typing import Dict, Any, List, Optional

from evaluate.unified_inference import UnifiedPredictor
from evaluate.explainability import ExplainabilityEngine
from evaluate.robustness import RobustnessEngine
from evaluate.absa import AspectBasedSentimentEngine
from backend.services.batch_service import BatchAnalyticsService

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Singleton service wrapping UnifiedPredictor and advanced analytics engines.
    """
    def __init__(self):
        self._predictor: Optional[UnifiedPredictor] = None
        self._is_loaded = False
        self.explain_engine: Optional[ExplainabilityEngine] = None
        self.robustness_engine: Optional[RobustnessEngine] = None
        self.absa_engine: Optional[AspectBasedSentimentEngine] = None
        self.batch_service: Optional[BatchAnalyticsService] = None
        self._benchmark_cache: Optional[Dict[str, Any]] = None

    def load_models(self):
        """Initializes all models and engines into memory."""
        if self._is_loaded:
            return
        
        logger.info("[InferenceService] Initializing UnifiedPredictor on GPU/CPU...")
        self._predictor = UnifiedPredictor()
        
        # Instantiate secondary analysis engines
        self.explain_engine = ExplainabilityEngine(self._predictor)
        self.robustness_engine = RobustnessEngine(self._predictor)
        
        # Best model for ABSA
        best_predictor = (
            self._predictor.predictors.get("DistilBERT") or 
            self._predictor.predictors.get("SVM") or 
            self._predictor.predictors.get("Logistic Regression")
        )
        self.absa_engine = AspectBasedSentimentEngine(best_predictor)
        self.batch_service = BatchAnalyticsService(self._predictor)
        
        self._is_loaded = True
        logger.info("[InferenceService] All models and analytics engines successfully loaded.")

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
        """Runs the text through all models."""
        if not self._is_loaded or not self._predictor:
            raise RuntimeError("Models are not loaded. Service is unavailable.")

        t0 = time.perf_counter()
        predictions = []
        errors = []

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

    def get_explanation(self, text: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Token attribution heatmap generator."""
        if not self.explain_engine:
            raise RuntimeError("Explainability engine is not initialized.")
        return self.explain_engine.explain(text, model_name)

    def get_aspect_analysis(self, text: str) -> Dict[str, Any]:
        """Aspect-Based Sentiment Analysis."""
        if not self.absa_engine:
            raise RuntimeError("ABSA engine is not initialized.")
        return self.absa_engine.extract_aspects(text)

    def run_stress_test(self, text: str) -> Dict[str, Any]:
        """Dynamic perturbation stress-test for a given input."""
        if not self.robustness_engine:
            raise RuntimeError("Robustness engine is not initialized.")
        return self.robustness_engine.stress_test_input(text)

    def get_robustness_benchmark(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Runs or returns cached benchmark results for the challenge suite."""
        if not self.robustness_engine:
            raise RuntimeError("Robustness engine is not initialized.")
        if self._benchmark_cache is None or force_refresh:
            self._benchmark_cache = self.robustness_engine.run_benchmark_suite()
        return self._benchmark_cache

    def process_batch_reviews(self, texts: List[str], primary_model: str = "DistilBERT") -> Dict[str, Any]:
        """Batch analytics across review texts."""
        if not self.batch_service:
            raise RuntimeError("Batch service is not initialized.")
        return self.batch_service.process_reviews(texts, primary_model=primary_model)

    def process_batch_csv(self, file_bytes: bytes, primary_model: str = "DistilBERT") -> Dict[str, Any]:
        """Batch analytics from uploaded CSV bytes."""
        if not self.batch_service:
            raise RuntimeError("Batch service is not initialized.")
        return self.batch_service.process_csv_file(file_bytes, primary_model=primary_model)

    def get_sample_batch(self) -> Dict[str, Any]:
        """Returns pre-loaded sample Amazon reviews analysis."""
        if not self.batch_service:
            raise RuntimeError("Batch service is not initialized.")
        return self.batch_service.get_sample_dataset()


# Instantiate singleton instance
inference_service = InferenceService()
