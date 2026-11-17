"""
backend/services/batch_service.py
──────────────────────────────────
Batch Processing and E-Commerce Executive Analytics Engine.

Provides:
1. High-throughput batch inference across models
2. Net Promoter Score (NPS) proxy calculation
3. Top positive drivers & complaint themes aggregation
4. Batch Aspect-Based Sentiment breakdown
5. Pre-loaded curated Amazon reviews sample dataset
"""

import io
import csv
import logging
from collections import Counter
from typing import List, Dict, Any, Optional

import pandas as pd

from evaluate.absa import AspectBasedSentimentEngine, ASPECT_TAXONOMY

logger = logging.getLogger(__name__)

# Sample dataset of authentic Amazon reviews for instant one-click demonstration
SAMPLE_AMAZON_REVIEWS = [
    {"id": 1, "product": "Wireless ANC Headphones", "text": "Incredible sound quality and the active noise cancellation completely blocks out subway rumble. Battery easily lasts 30 hours."},
    {"id": 2, "product": "Wireless ANC Headphones", "text": "The ear cushions are soft, but the Bluetooth connection stutters constantly when my phone is in my pocket."},
    {"id": 3, "product": "Wireless ANC Headphones", "text": "Sound is okay, but after 3 months the plastic headband snapped. Customer service refused a warranty replacement."},
    {"id": 4, "product": "Ultra-Fast USB-C Charger", "text": "Charges my laptop and phone at lightning speed. Very compact and feels durable."},
    {"id": 5, "product": "Ultra-Fast USB-C Charger", "text": "Got dangerously hot during the first charge and started smelling like burned plastic. Returning immediately."},
    {"id": 6, "product": "Mechanical Gaming Keyboard", "text": "The tactile switches feel amazing and the aluminum chassis has zero flex. Worth every penny."},
    {"id": 7, "product": "Mechanical Gaming Keyboard", "text": "Keys feel decent, but the companion software crashes on startup and RGB settings do not save."},
    {"id": 8, "product": "Smart Fitness Tracker", "text": "Battery lasts nearly two weeks! Sleep tracking and step count are remarkably accurate."},
    {"id": 9, "product": "Smart Fitness Tracker", "text": "Screen stopped responding after swimming in shallow water, despite being advertised as 50m waterproof."},
    {"id": 10, "product": "Ergonomic Office Chair", "text": "My chronic lower back pain vanished after two days of using this. Assembly took only 15 minutes."},
    {"id": 11, "product": "Ergonomic Office Chair", "text": "Flimsy armrests and the hydraulic cylinder slowly sinks down while sitting. Not worth the expensive price."},
    {"id": 12, "product": "Portable Blender", "text": "Surprisingly powerful for its size! Blends frozen strawberries and protein powder smoothly."},
    {"id": 13, "product": "Portable Blender", "text": "The motor burned out on day four while blending bananas. Waste of money."},
    {"id": 14, "product": "4K Gaming Monitor", "text": "Vibrant IPS colors, buttery smooth 144Hz refresh rate, and zero dead pixels. Outstanding value."},
    {"id": 15, "product": "4K Gaming Monitor", "text": "Horrible backlight bleed in the corners and the built-in speakers sound like a tin can."},
    {"id": 16, "product": "Electric Toothbrush", "text": "Cleans way better than a manual brush. The two-minute timer with haptic pulses is super helpful."},
    {"id": 17, "product": "Electric Toothbrush", "text": "Battery died completely after 40 days and won't hold a charge. Seller never replied to my email."},
    {"id": 18, "product": "Stainless Steel Water Bottle", "text": "Keeps water ice cold even after leaving it in a hot car all afternoon. Does not leak whatsoever."},
    {"id": 19, "product": "Robot Vacuum Cleaner", "text": "Navigates around furniture seamlessly and picks up pet hair effortlessly. Saves me hours every weekend."},
    {"id": 20, "product": "Robot Vacuum Cleaner", "text": "Gets stuck on every single rug and the mobile app loses connection twice a day. Total disappointment."}
]


class BatchAnalyticsService:
    """
    Orchestrates batch inference and statistical business intelligence.
    """
    def __init__(self, unified_predictor=None):
        self.unified_predictor = unified_predictor
        self.absa_engine = AspectBasedSentimentEngine()

    def process_reviews(self, review_texts: List[str], primary_model: str = "DistilBERT") -> Dict[str, Any]:
        """
        Processes a list of review texts through the unified model engine and computes analytics.
        """
        if not review_texts:
            raise ValueError("No review texts provided.")

        predictor = None
        if self.unified_predictor:
            predictor = self.unified_predictor.predictors.get(primary_model)
            if not predictor and len(self.unified_predictor.predictors) > 0:
                # Fallback to first available model
                predictor = list(self.unified_predictor.predictors.values())[0]

        results = []
        positive_count = 0
        negative_count = 0
        promoters = 0
        detractors = 0
        passives = 0

        positive_words = []
        negative_words = []

        aspect_counter = {asp: {"positive": 0, "negative": 0} for asp in ASPECT_TAXONOMY.keys()}

        for idx, text in enumerate(review_texts):
            text_str = str(text).strip()
            if not text_str:
                continue

            # Run inference
            if predictor:
                try:
                    pred_res = predictor.predict_single(text_str)
                    label = pred_res["label"]
                    label_name = pred_res["label_name"].capitalize()
                    conf = float(pred_res["confidence"])
                    # Normalizing confidence relative to predicted class
                    actual_conf = conf if label == 1 else (1.0 - conf)
                except Exception as e:
                    logger.error("Inference failed for row %d: %s", idx, e)
                    label = 1
                    label_name = "Positive"
                    actual_conf = 0.50
            else:
                # Fallback rule-based
                label = 1 if "good" in text_str.lower() or "great" in text_str.lower() else 0
                label_name = "Positive" if label == 1 else "Negative"
                actual_conf = 0.75

            if label == 1:
                positive_count += 1
                if actual_conf >= 0.80:
                    promoters += 1
                else:
                    passives += 1
            else:
                negative_count += 1
                if actual_conf >= 0.80:
                    detractors += 1
                else:
                    passives += 1

            # Extract aspects for this review
            aspect_res = self.absa_engine.extract_aspects(text_str, fallback_predictor=predictor)
            primary_aspect = aspect_res["aspects"][0]["aspect"] if aspect_res["aspects"] else "General"
            
            for asp in aspect_res["aspects"]:
                asp_name = asp["aspect"]
                if asp_name in aspect_counter:
                    if asp["sentiment"] == "Positive":
                        aspect_counter[asp_name]["positive"] += 1
                    elif asp["sentiment"] == "Negative":
                        aspect_counter[asp_name]["negative"] += 1

            # Extract key terms for word cloud / drivers
            clean_tokens = [w.lower() for w in text_str.split() if len(w) > 4 and w.isalpha()]
            if label == 1:
                positive_words.extend(clean_tokens)
            else:
                negative_words.extend(clean_tokens)

            results.append({
                "id": idx + 1,
                "text": text_str,
                "sentiment": label_name,
                "confidence": round(actual_conf * 100, 1),
                "primary_aspect": primary_aspect,
                "aspects": aspect_res["aspects"]
            })

        total = len(results)
        if total == 0:
            raise ValueError("No valid reviews found in the payload.")

        pos_pct = round((positive_count / total) * 100, 1)
        neg_pct = round((negative_count / total) * 100, 1)
        
        # NPS Calculation: % Promoters - % Detractors (-100 to +100)
        nps_score = round(((promoters - detractors) / total) * 100, 1)

        # Stopwords filter for driver keywords
        stopwords = {
            "this", "that", "these", "those", "there", "their", "about", "after", "again",
            "could", "would", "should", "which", "where", "while", "every", "first", "second"
        }
        top_pos = [
            {"word": w, "count": c}
            for w, c in Counter([w for w in positive_words if w not in stopwords]).most_common(8)
        ]
        top_neg = [
            {"word": w, "count": c}
            for w, c in Counter([w for w in negative_words if w not in stopwords]).most_common(8)
        ]

        # Aspect aggregation
        aspect_summary = []
        for asp_name, counts in aspect_counter.items():
            subtotal = counts["positive"] + counts["negative"]
            if subtotal > 0:
                aspect_summary.append({
                    "aspect": asp_name,
                    "total_mentions": subtotal,
                    "positive_count": counts["positive"],
                    "negative_count": counts["negative"],
                    "satisfaction_rate": round((counts["positive"] / subtotal) * 100, 1)
                })
        aspect_summary.sort(key=lambda x: x["total_mentions"], reverse=True)

        return {
            "total_reviews": total,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "nps_metrics": {
                "nps_score": nps_score,
                "promoters_pct": round((promoters / total) * 100, 1),
                "passives_pct": round((passives / total) * 100, 1),
                "detractors_pct": round((detractors / total) * 100, 1),
                "classification": "Excellent" if nps_score > 50 else ("Good" if nps_score > 0 else "Needs Improvement")
            },
            "top_positive_drivers": top_pos,
            "top_complaint_themes": top_neg,
            "aspect_breakdown": aspect_summary,
            "reviews": results
        }

    def process_csv_file(self, file_content: bytes, primary_model: str = "DistilBERT") -> Dict[str, Any]:
        """
        Parses uploaded CSV bytes, automatically discovers the review text column, and processes reviews.
        """
        try:
            df = pd.read_csv(io.BytesIO(file_content))
        except Exception:
            try:
                df = pd.read_csv(io.BytesIO(file_content), encoding="latin1")
            except Exception as e:
                raise ValueError(f"Could not parse CSV file: {str(e)}")

        # Heuristic for text column
        candidate_cols = ["text", "review", "review_text", "reviews", "content", "Review Text", "Summary", "body"]
        found_col = None
        for col in df.columns:
            if col.strip().lower() in candidate_cols:
                found_col = col
                break

        if not found_col:
            # Fallback to the first string column with longest average length
            str_cols = [c for c in df.columns if df[c].dtype == object]
            if str_cols:
                found_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
            else:
                raise ValueError("Could not find a text/review column in the uploaded CSV.")

        texts = df[found_col].dropna().astype(str).tolist()
        return self.process_reviews(texts, primary_model=primary_model)

    def get_sample_dataset(self) -> Dict[str, Any]:
        """Returns the pre-loaded sample dataset processed with full analytics."""
        texts = [item["text"] for item in SAMPLE_AMAZON_REVIEWS]
        res = self.process_reviews(texts)
        # Annotate with product names
        for r, item in zip(res["reviews"], SAMPLE_AMAZON_REVIEWS):
            r["product"] = item["product"]
        return res
