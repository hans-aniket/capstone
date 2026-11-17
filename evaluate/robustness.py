"""
evaluate/robustness.py
──────────────────────
Adversarial, Sarcasm, and Linguistic Robustness Testing Engine (CheckList framework).

Evaluates model resilience across 5 core stress vectors:
1. Negation & Context Reversal ("not bad at all" vs "not good")
2. Sarcasm & Irony ("Oh wonderful, broke after 2 hours")
3. Contrastive Conjunctions ("Great packaging, but defective product")
4. Character Noise & Keyboard Typos ("amaaaazing", "defctve")
5. Length Extremes (1-word reviews vs multi-paragraph)
"""

import random
import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Fixed seed for reproducibility in benchmark tests
_RANDOM_SEED = 42

# Curated benchmark challenge suite with verified ground-truth labels (0=negative, 1=positive)
BENCHMARK_CHALLENGE_SUITE: List[Dict[str, Any]] = [
    # ── 1. Negation & Reversals ───────────────────────────────────────────────
    {
        "category": "Negation Handling",
        "text": "The battery life is not bad at all, lasts the entire day.",
        "expected_label": 1,
        "difficulty": "medium",
        "description": "Double negative / litotes ('not bad at all')"
    },
    {
        "category": "Negation Handling",
        "text": "I would hardly call this a failure; it exceeded my expectations.",
        "expected_label": 1,
        "difficulty": "hard",
        "description": "Nuanced negation ('hardly call this a failure')"
    },
    {
        "category": "Negation Handling",
        "text": "This is definitely not the quality I was hoping for.",
        "expected_label": 0,
        "difficulty": "medium",
        "description": "Direct negation of positive attribute ('not the quality')"
    },
    {
        "category": "Negation Handling",
        "text": "By no means is this product worth the price tag.",
        "expected_label": 0,
        "difficulty": "hard",
        "description": "Fronted negative adverbial ('By no means')"
    },
    {
        "category": "Negation Handling",
        "text": "Never fails to impress me every time I use it.",
        "expected_label": 1,
        "difficulty": "hard",
        "description": "Idiomatic double negation ('Never fails to impress')"
    },

    # ── 2. Sarcasm & Irony ────────────────────────────────────────────────────
    {
        "category": "Sarcasm & Irony",
        "text": "Oh fantastic, it completely stopped working after 2 hours.",
        "expected_label": 0,
        "difficulty": "hard",
        "description": "Sarcastic positive exclamation followed by immediate defect"
    },
    {
        "category": "Sarcasm & Irony",
        "text": "Brilliant engineering if your intention was to create a paperweight.",
        "expected_label": 0,
        "difficulty": "hard",
        "description": "Ironical compliment ('paperweight')"
    },
    {
        "category": "Sarcasm & Irony",
        "text": "What a delightful waste of fifty dollars.",
        "expected_label": 0,
        "difficulty": "medium",
        "description": "Oxymoronic adjective-noun pairing ('delightful waste')"
    },
    {
        "category": "Sarcasm & Irony",
        "text": "Just what I always wanted, a phone charger that sparks and overheats.",
        "expected_label": 0,
        "difficulty": "hard",
        "description": "Pretended enthusiasm ('Just what I always wanted')"
    },
    {
        "category": "Sarcasm & Irony",
        "text": "10/10 would love to get electrocuted by this again.",
        "expected_label": 0,
        "difficulty": "hard",
        "description": "Satirical rating trope"
    },

    # ── 3. Contrastive Conjunctions (Ambivalence) ─────────────────────────────
    {
        "category": "Contrastive Conjunctions",
        "text": "The packaging was exquisite and pristine, but the actual device arrived dead.",
        "expected_label": 0,
        "difficulty": "medium",
        "description": "Positive preamble negated by pivotal 'but'"
    },
    {
        "category": "Contrastive Conjunctions",
        "text": "Delivery was frustratingly delayed, however the headphones sound extraordinary.",
        "expected_label": 1,
        "difficulty": "medium",
        "description": "Negative preamble redeemed by pivotal 'however'"
    },
    {
        "category": "Contrastive Conjunctions",
        "text": "Although the interface feels slightly clunky, the speed and accuracy are unmatched.",
        "expected_label": 1,
        "difficulty": "medium",
        "description": "Subordinate concession ('Although ...') followed by strong core"
    },
    {
        "category": "Contrastive Conjunctions",
        "text": "Despite glowing reviews, the build quality feels shockingly cheap and flimsy.",
        "expected_label": 0,
        "difficulty": "medium",
        "description": "Contrastive preposition ('Despite glowing reviews')"
    },

    # ── 4. Typos & Noise Perturbations ────────────────────────────────────────
    {
        "category": "Typos & Noise",
        "text": "Absooolutely amaaaazing quality, looooved it so much!",
        "expected_label": 1,
        "difficulty": "easy",
        "description": "Character repetitions / emotional elongation"
    },
    {
        "category": "Typos & Noise",
        "text": "Terible prodcut. Brokn on arival, waste of mony.",
        "expected_label": 0,
        "difficulty": "medium",
        "description": "Multiple misspellings and dropped letters"
    },
    {
        "category": "Typos & Noise",
        "text": "Defective item!! DO NOT BUY!! SCAM!!!",
        "expected_label": 0,
        "difficulty": "easy",
        "description": "All-caps emphasis with aggressive punctuation"
    },
    {
        "category": "Typos & Noise",
        "text": "Prdct works gr8, grt val u 4 mny.",
        "expected_label": 1,
        "difficulty": "hard",
        "description": "Heavy leetspeak and abbreviated text"
    },

    # ── 5. Extreme Length ─────────────────────────────────────────────────────
    {
        "category": "Length Extremes",
        "text": "Horrendous.",
        "expected_label": 0,
        "difficulty": "medium",
        "description": "Single-word negative review"
    },
    {
        "category": "Length Extremes",
        "text": "Flawless.",
        "expected_label": 1,
        "difficulty": "medium",
        "description": "Single-word positive review"
    },
    {
        "category": "Length Extremes",
        "text": "I have been evaluating and benchmarking consumer electronics for over fifteen years across multiple commercial domains and testing labs. In all my years of detailed inspection, rarely have I encountered an item built with such exquisite attention to detail, impeccable tolerances, and consistent reliability.",
        "expected_label": 1,
        "difficulty": "easy",
        "description": "Long formal multi-clause review"
    }
]


class RobustnessEngine:
    """
    Evaluates models on linguistic stress tests and generates on-the-fly perturbations.
    """
    def __init__(self, unified_predictor=None):
        self.unified_predictor = unified_predictor

    # ── Dynamic Perturbation Generators ───────────────────────────────────────

    @staticmethod
    def generate_typos(text: str, severity: float = 0.2) -> str:
        """Introduce character swaps and keyboard proximity typos."""
        words = text.split()
        if not words:
            return text

        perturbed = []
        for word in words:
            if len(word) > 4 and random.random() < severity:
                # Character swap
                idx = random.randint(1, len(word) - 2)
                chars = list(word)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                perturbed.append("".join(chars))
            elif len(word) > 3 and random.random() < severity:
                # Elongate character (e.g. good -> gooood)
                idx = random.randint(0, len(word) - 1)
                chars = list(word)
                chars[idx] = chars[idx] * 3
                perturbed.append("".join(chars))
            else:
                perturbed.append(word)
        return " ".join(perturbed)

    @staticmethod
    def inject_sarcasm(text: str) -> str:
        """Wrap text in sarcastic framing."""
        sarcastic_intros = [
            "Oh absolutely wonderful, ",
            "Just brilliant... ",
            "What a masterclass in perfection: ",
            "Truly marvelous how "
        ]
        return random.choice(sarcastic_intros) + text.lower()

    @staticmethod
    def inject_negation_flip(text: str) -> str:
        """Inject negation words to stress-test context handling."""
        if "not" in text.lower():
            return re.sub(r"\bnot\s+", "", text, flags=re.IGNORECASE)
        else:
            words = text.split()
            if len(words) > 2:
                words.insert(2, "not")
                return " ".join(words)
            return "It is not " + text

    def stress_test_input(self, text: str) -> Dict[str, Any]:
        """
        Takes a single input text, produces 3 perturbed variants,
        and tests all loaded models across each variant to inspect stability.
        """
        variants = [
            {
                "type": "Original",
                "text": text,
                "description": "Baseline unperturbed input"
            },
            {
                "type": "Noise / Typos",
                "text": self.generate_typos(text, severity=0.35),
                "description": "Character swaps & typos injected"
            },
            {
                "type": "Sarcastic Framing",
                "text": self.inject_sarcasm(text),
                "description": "Ironical / sarcastic prefix applied"
            },
            {
                "type": "Negation Perturbation",
                "text": self.inject_negation_flip(text),
                "description": "Syntactic negation added/removed"
            }
        ]

        results = []
        for v in variants:
            v_text = v["text"]
            model_preds = {}
            if self.unified_predictor:
                for name, pred in self.unified_predictor.predictors.items():
                    try:
                        res = pred.predict_single(v_text)
                        model_preds[name] = {
                            "label": res["label_name"],
                            "confidence": round(res["confidence"], 4)
                        }
                    except Exception as e:
                        model_preds[name] = {"error": str(e)}
            
            results.append({
                "perturbation_type": v["type"],
                "description": v["description"],
                "text": v_text,
                "predictions": model_preds
            })

        return {
            "original_text": text,
            "perturbation_count": len(results),
            "results": results
        }

    # ── Benchmark Suite Evaluation ────────────────────────────────────────────

    def run_benchmark_suite(self) -> Dict[str, Any]:
        """
        Runs the full challenge test suite across all loaded models.
        Returns accuracy and resilience per category and overall leaderboard.
        """
        if not self.unified_predictor:
            raise RuntimeError("UnifiedPredictor is required to run benchmark suite.")

        categories = sorted(list(set(case["category"] for case in BENCHMARK_CHALLENGE_SUITE)))
        models = list(self.unified_predictor.predictors.keys())

        # Track score per model per category
        category_stats = {cat: {m: {"correct": 0, "total": 0} for m in models} for cat in categories}
        model_overall = {m: {"correct": 0, "total": 0} for m in models}
        detailed_cases = []

        for case in BENCHMARK_CHALLENGE_SUITE:
            text = case["text"]
            target = case["expected_label"]
            target_name = "positive" if target == 1 else "negative"
            cat = case["category"]

            case_preds = {}
            for m in models:
                predictor = self.unified_predictor.predictors[m]
                try:
                    res = predictor.predict_single(text)
                    is_correct = (res["label"] == target)
                    if is_correct:
                        category_stats[cat][m]["correct"] += 1
                        model_overall[m]["correct"] += 1
                    category_stats[cat][m]["total"] += 1
                    model_overall[m]["total"] += 1

                    case_preds[m] = {
                        "prediction": res["label_name"],
                        "confidence": round(res["confidence"], 4),
                        "correct": is_correct
                    }
                except Exception as e:
                    case_preds[m] = {"error": str(e), "correct": False}

            detailed_cases.append({
                "category": cat,
                "text": text,
                "expected": target_name,
                "difficulty": case["difficulty"],
                "description": case["description"],
                "predictions": case_preds
            })

        # Format category leaderboard
        category_summary = []
        for cat in categories:
            cat_row = {"category": cat, "models": {}}
            for m in models:
                tot = category_stats[cat][m]["total"]
                corr = category_stats[cat][m]["correct"]
                acc = round((corr / tot) * 100, 1) if tot > 0 else 0.0
                cat_row["models"][m] = {
                    "accuracy_pct": acc,
                    "correct": corr,
                    "total": tot
                }
            category_summary.append(cat_row)

        # Overall leaderboard
        leaderboard = []
        for m in models:
            tot = model_overall[m]["total"]
            corr = model_overall[m]["correct"]
            acc = round((corr / tot) * 100, 1) if tot > 0 else 0.0
            
            # Determine tier
            tier = getattr(self.unified_predictor.predictors[m], "tier", "unknown")
            leaderboard.append({
                "model": m,
                "tier": tier,
                "robustness_score": acc,
                "passed": corr,
                "total": tot
            })

        leaderboard.sort(key=lambda x: x["robustness_score"], reverse=True)

        return {
            "total_test_cases": len(BENCHMARK_CHALLENGE_SUITE),
            "categories": categories,
            "leaderboard": leaderboard,
            "category_breakdown": category_summary,
            "cases": detailed_cases
        }
