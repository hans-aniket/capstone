"""
evaluate/absa.py
────────────────
Aspect-Based Sentiment Analysis (ABSA) Engine.

Extracts fine-grained opinions on specific product dimensions:
- Battery & Charging
- Build Quality & Durability
- Performance & Speed
- Price & Value
- Design & Ergonomics
- Customer Support & Delivery
- Sound & Display / Media
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Aspect ontology tailored for consumer electronics & Amazon product reviews
ASPECT_TAXONOMY = {
    "Battery & Charging": [
        "battery", "batteries", "charge", "charger", "charging", "recharge",
        "power", "mah", "cord", "drain", "drains", "draining", "runtime"
    ],
    "Build Quality & Durability": [
        "quality", "build", "material", "plastic", "metal", "durable", "durability",
        "sturdy", "solid", "break", "broke", "broken", "flimsy", "cheap", "fragile"
    ],
    "Performance & Speed": [
        "performance", "speed", "fast", "slow", "lag", "laggy", "smooth", "responsive",
        "freeze", "crash", "powerful", "processor", "ram", "glitch", "work", "works"
    ],
    "Price & Value": [
        "price", "cost", "value", "money", "expensive", "cheap", "worth", "bargain",
        "deal", "dollar", "dollars", "penny", "overpriced", "affordable"
    ],
    "Design & Ergonomics": [
        "design", "look", "looks", "style", "comfortable", "comfort", "ergonomic",
        "weight", "heavy", "light", "compact", "sleek", "color", "fit", "size"
    ],
    "Customer Support & Delivery": [
        "service", "customer service", "shipping", "delivery", "arrived", "seller",
        "return", "refund", "warranty", "package", "packaging", "box", "support"
    ],
    "Sound & Display / Media": [
        "sound", "audio", "speaker", "speakers", "volume", "bass", "clarity",
        "screen", "display", "resolution", "bright", "brightness", "picture", "color"
    ]
}


class AspectBasedSentimentEngine:
    """
    Extracts aspect-level sentiment by isolating syntactic clauses mentioning target aspects.
    """
    def __init__(self, predictor=None):
        self.predictor = predictor

    def _split_into_clauses(self, text: str) -> List[str]:
        """Split text into sentences and conjunction-bounded clauses."""
        # Split on sentence boundaries and contrastive conjunctions
        raw_splits = re.split(r'[.!?;\n]|(?:\b(?:but|however|although|though|yet|whereas|and)\b)', text, flags=re.IGNORECASE)
        clauses = [c.strip() for c in raw_splits if len(c.strip()) > 3]
        return clauses if clauses else [text.strip()]

    def extract_aspects(self, text: str, fallback_predictor=None) -> Dict[str, Any]:
        """
        Identify aspects mentioned in the text, extract corresponding clauses,
        and predict sentiment polarity for each identified aspect.
        """
        predictor = self.predictor or fallback_predictor
        clauses = self._split_into_clauses(text)
        aspect_opinions = []
        seen_aspects = set()

        text_lower = text.lower()

        for aspect_name, keywords in ASPECT_TAXONOMY.items():
            # Check if aspect is mentioned in the review
            matching_keywords = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]
            if not matching_keywords:
                continue

            # Find the most relevant clause for this aspect
            relevant_clauses = []
            for clause in clauses:
                c_lower = clause.lower()
                if any(re.search(r'\b' + re.escape(kw) + r'\b', c_lower) for kw in matching_keywords):
                    relevant_clauses.append(clause)

            snippet = " ".join(relevant_clauses) if relevant_clauses else text

            # Evaluate sentiment on the isolated snippet
            sentiment = "Neutral"
            confidence = 0.50
            polarity_score = 0.0

            if predictor:
                try:
                    res = predictor.predict_single(snippet)
                    p_pos = float(res["confidence"])
                    if res["label"] == 1:
                        sentiment = "Positive"
                        confidence = round(p_pos, 3)
                        polarity_score = round(p_pos, 3)
                    else:
                        sentiment = "Negative"
                        confidence = round(1.0 - p_pos, 3)
                        polarity_score = round(-(1.0 - p_pos), 3)
                except Exception as e:
                    logger.warning("Aspect prediction failed for %s: %s", aspect_name, e)

            # Heuristic adjustment for strong negation in snippet if model missed it
            if re.search(r'\b(?:not|no|hardly|never|terrible|awful|horrible|broken|worst)\b', snippet.lower()):
                if sentiment == "Positive" and confidence < 0.80:
                    sentiment = "Negative"
                    polarity_score = -0.75
                    confidence = 0.75

            aspect_opinions.append({
                "aspect": aspect_name,
                "keywords_matched": matching_keywords[:3],
                "sentiment": sentiment,
                "confidence": confidence,
                "polarity_score": polarity_score,
                "snippet": snippet[:150] + ("..." if len(snippet) > 150 else "")
            })
            seen_aspects.add(aspect_name)

        # Overall summary of aspects
        pos_count = sum(1 for a in aspect_opinions if a["sentiment"] == "Positive")
        neg_count = sum(1 for a in aspect_opinions if a["sentiment"] == "Negative")

        if pos_count > 0 and neg_count > 0:
            dominant = "Mixed"
        elif pos_count > 0:
            dominant = "Predominantly Positive"
        elif neg_count > 0:
            dominant = "Predominantly Negative"
        else:
            dominant = "Neutral / No Specific Aspects"

        return {
            "aspects_found": len(aspect_opinions),
            "dominant_tone": dominant,
            "aspects": aspect_opinions
        }
