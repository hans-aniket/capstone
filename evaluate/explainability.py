"""
evaluate/explainability.py
──────────────────────────
Model Explainability (XAI) engine for Comparative Sentiment Analysis.

Supports:
1. Classical Models (Logistic Regression, Linear SVM, Naive Bayes):
   - Computes feature contribution = TF-IDF weight * classifier coefficient / log-ratio.
2. Transformer Models (DistilBERT):
   - Computes gradient-based token attribution (Input x Gradient) on GPU.
3. Unified interface returning token-level highlights for React frontend heatmap.
"""

import re
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch

from models.classical.predictor import ClassicalPredictor
from preprocessing.classical_preprocessor import ClassicalPreprocessor

logger = logging.getLogger(__name__)

# Stopwords to keep attribution focused on meaningful sentiment markers
_COMMON_PUNCT = set(".,!?:;\"'()[]{}-/\\")


class ExplainabilityEngine:
    """
    Computes token-level attributions and saliency maps for sentiment predictions.
    """
    def __init__(self, unified_predictor=None):
        self.unified_predictor = unified_predictor

    def explain_classical(self, text: str, model_key: str = "logistic_regression") -> Dict[str, Any]:
        """
        Explain classical linear model by multiplying TF-IDF feature value with model weight.
        """
        if self.unified_predictor and model_key in self.unified_predictor.predictors:
            predictor = self.unified_predictor.predictors[model_key]
        else:
            try:
                predictor = ClassicalPredictor.from_saved(model_key)
            except Exception as e:
                logger.error("Could not load predictor for %s: %s", model_key, e)
                return {"tokens": [], "error": str(e)}

        clf = getattr(predictor.trainer, "model", None)
        preprocessor: ClassicalPreprocessor = predictor.preprocessor
        vectorizer = preprocessor.vectorizer

        # Tokenize preserving original words
        raw_words = re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text)
        feature_names = vectorizer.get_feature_names_out()
        vocab_index = vectorizer.vocabulary_

        # Get weights
        if hasattr(clf, "coef_"):
            coefs = clf.coef_[0]
        elif hasattr(clf, "feature_log_prob_"):
            # Naive Bayes: difference of log probabilities
            coefs = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        elif hasattr(clf, "calibrated_classifiers_"):
            # CalibratedClassifierCV (SVM)
            coefs = np.mean([cc.estimator.coef_[0] for cc in clf.calibrated_classifiers_], axis=0)
        else:
            coefs = np.zeros(len(feature_names))

        # Vectorize single document
        tfidf_vec = vectorizer.transform([text])
        non_zeros = dict(zip(tfidf_vec.indices, tfidf_vec.data))

        # Token contributions
        token_attributions = []
        max_score = 1e-6

        for word in raw_words:
            norm_word = word.lower()
            if norm_word in vocab_index and vocab_index[norm_word] in non_zeros:
                idx = vocab_index[norm_word]
                score = float(non_zeros[idx] * coefs[idx])
            else:
                score = 0.0

            if abs(score) > max_score:
                max_score = abs(score)

            sentiment = "neutral"
            if score > 0.01:
                sentiment = "positive"
            elif score < -0.01:
                sentiment = "negative"

            token_attributions.append({
                "word": word,
                "score": score,
                "sentiment": sentiment,
                "importance": 0.0  # normalized later
            })

        # Normalize importance 0 to 1
        for item in token_attributions:
            item["importance"] = round(abs(item["score"]) / max_score, 4)
            item["score"] = round(item["score"], 4)

        pred_res = predictor.predict_single(text)
        
        # Sort words by positive and negative drivers
        sorted_pos = sorted(
            [t for t in token_attributions if t["sentiment"] == "positive"],
            key=lambda x: x["score"],
            reverse=True
        )
        sorted_neg = sorted(
            [t for t in token_attributions if t["sentiment"] == "negative"],
            key=lambda x: x["score"]
        )

        return {
            "model": predictor.model_name,
            "tier": "classical",
            "predicted_label": pred_res["label_name"],
            "confidence": pred_res["confidence"],
            "tokens": token_attributions,
            "top_positive": [t["word"] for t in sorted_pos[:5]],
            "top_negative": [t["word"] for t in sorted_neg[:5]]
        }

    def explain_transformer(self, text: str) -> Dict[str, Any]:
        """
        Explain DistilBERT prediction using Input x Gradient saliency on GPU.
        """
        if not self.unified_predictor or "DistilBERT" not in self.unified_predictor.predictors:
            # Fallback to classical if DistilBERT is not loaded
            return self.explain_classical(text, "logistic_regression")

        predictor = self.unified_predictor.predictors["DistilBERT"]
        model = predictor.model
        tokenizer = predictor.preprocessor.tokenizer
        device = predictor.device

        model.eval()
        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            return_offsets_mapping=True
        )
        
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        offset_mapping = tokens["offset_mapping"][0].cpu().numpy()

        # Get embeddings with gradient tracking
        embeddings = model.distilbert.embeddings(input_ids)
        embeddings.retain_grad()

        # Direct forward pass using inputs_embeds
        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        
        probs = torch.softmax(logits, dim=-1)
        pred_label_idx = torch.argmax(logits, dim=-1).item()
        target_score = logits[0, pred_label_idx]

        model.zero_grad()
        target_score.backward()

        # Input x Gradient saliency: dot product of embedding with grad
        if embeddings.grad is not None:
            saliency = (embeddings.grad * embeddings).sum(dim=-1).squeeze(0)  # (seq_len,)
            saliency = saliency.detach().cpu().numpy()
        else:
            saliency = np.zeros(input_ids.shape[1])

        # If predicted class is 0 (negative), invert saliency so positive drivers are + and negative are -
        multiplier = 1.0 if pred_label_idx == 1 else -1.0
        saliency = saliency * multiplier

        token_list = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        
        # Merge subwords / assemble token attributions
        token_attributions = []
        max_s = float(np.max(np.abs(saliency))) if len(saliency) > 0 and np.max(np.abs(saliency)) > 0 else 1.0

        for tok, score, (start, end) in zip(token_list, saliency, offset_mapping):
            if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            word = text[start:end] if (start != end and start < len(text)) else tok.replace("##", "")
            if not word.strip():
                continue

            sentiment = "neutral"
            if score > 0.05 * max_s:
                sentiment = "positive"
            elif score < -0.05 * max_s:
                sentiment = "negative"

            token_attributions.append({
                "word": word,
                "score": round(float(score), 4),
                "sentiment": sentiment,
                "importance": round(float(abs(score) / max_s), 4)
            })

        pred_res = predictor.predict_single(text)
        sorted_pos = sorted(
            [t for t in token_attributions if t["sentiment"] == "positive"],
            key=lambda x: x["score"],
            reverse=True
        )
        sorted_neg = sorted(
            [t for t in token_attributions if t["sentiment"] == "negative"],
            key=lambda x: x["score"]
        )

        return {
            "model": "DistilBERT",
            "tier": "transformers",
            "predicted_label": pred_res["label_name"],
            "confidence": pred_res["confidence"],
            "tokens": token_attributions,
            "top_positive": [t["word"] for t in sorted_pos[:5]],
            "top_negative": [t["word"] for t in sorted_neg[:5]]
        }

    def explain(self, text: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point for explainability. Returns token attribution for requested model
        or defaults to DistilBERT (or SVM/Logistic Regression if transformer unavailable).
        """
        if not text or not text.strip():
            return {"error": "Empty text provided"}

        if model_name and "distilbert" in model_name.lower():
            return self.explain_transformer(text)
        elif model_name and "svm" in model_name.lower():
            return self.explain_classical(text, "svm")
        elif model_name and "bayes" in model_name.lower():
            return self.explain_classical(text, "naive_bayes")
        elif model_name and "logistic" in model_name.lower():
            return self.explain_classical(text, "logistic_regression")
        else:
            # Default to DistilBERT if available, else Logistic Regression
            if self.unified_predictor and "DistilBERT" in self.unified_predictor.predictors:
                return self.explain_transformer(text)
            return self.explain_classical(text, "logistic_regression")
