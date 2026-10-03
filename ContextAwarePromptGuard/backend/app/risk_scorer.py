"""
G2 Component: Context-Aware Benign Discriminator and Calibrated Risk Scorer.
Addresses Gap G2: Ambiguous benign instructions risk false positives.

When a benign input contains instructional phrases (e.g. "summarize the following",
"ignore spelling mistakes", or quoted commands in news/NLI premises), standard classifiers
often trigger false positives (e.g. 23.08% FPR on RTE clean samples in Phase 6).

This module provides:
1. Structural Context Discrimination:
   - Quotation containment analysis: determines if instructional phrases are enclosed in quotes.
   - Passive/Reported speech detection: identifies 3rd-person reported speech vs active imperatives.
   - Delimiter boundary containment: detects whether text respects or breaches structured wrappers.
2. Calibrated Continuous Risk Scoring (0–100):
   - Maps raw logits and context indicators via Platt scaling / temperature-calibrated sigmoid.
   - Classifies outputs into three actionable risk tiers:
     * 0 – 30: LOW_RISK (Benign prompt)
     * 31 – 70: MEDIUM_RISK (Ambiguous / suspicious pattern; sandboxing recommended)
     * 71 – 100: HIGH_RISK (Confirmed prompt injection attack)
"""

import re
import numpy as np


# Reporting / passive verbs indicating non-directive narrative text
REPORTING_VERBS = [
    r"\b(said|stated|reported|argued|claimed|noted|wrote|explained|mentioned|suggested)\b",
    r"\b(according to|quoted as saying|testified that|ruled that|voted to)\b",
    r"\b(was told to|were asked to|were instructed to|was ordered to)\b"
]
COMPILED_REPORTING = [re.compile(p, re.IGNORECASE) for p in REPORTING_VERBS]

# Quotation regex patterns
QUOTED_TEXT_REGEX = re.compile(r'["\'](.*?)["\']|“([^”]+)”|«([^»]+)»', re.DOTALL)

# True adversarial escape delimiters
ESCAPE_DELIMITERS = [
    r"```\s*(ignore|override|system|instruction)",
    r"---+\s*(new\s+task|instruction|prompt)",
    r"\[\s*system\s*\]|\{\s*system\s*\}|<system>",
    r"###\s*(instruction|system|human)",
    r"end\s+of\s+text\.\s*(now|instead|override)"
]
COMPILED_ESCAPE = [re.compile(p, re.IGNORECASE) for p in ESCAPE_DELIMITERS]


class CalibratedRiskScorer:
    def __init__(self, temperature=1.5, low_threshold=30, high_threshold=70):
        self.temperature = temperature
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def evaluate_quotation_safety(self, text):
        """
        Determines if imperative/directive phrases in the text are safely
        contained inside quotation marks (passive data) rather than commanding the LLM.
        """
        quotes = QUOTED_TEXT_REGEX.findall(text)
        flat_quotes = []
        for q_tuple in quotes:
            for q in q_tuple:
                if q:
                    flat_quotes.append(q)

        total_quote_chars = sum(len(q) for q in flat_quotes)
        quote_ratio = float(total_quote_chars / max(1, len(text)))

        # Check if directive words exist only inside quotes
        directive_words = ["ignore", "override", "bypass", "repeat", "output", "system", "rule"]
        outside_text = QUOTED_TEXT_REGEX.sub(" ", text)

        directives_outside = any(w in outside_text.lower() for w in directive_words)
        directives_inside = any(any(w in q.lower() for w in directive_words) for q in flat_quotes)

        # Safely quoted if directives are inside quotes but not outside
        is_safely_quoted = (directives_inside and not directives_outside and quote_ratio > 0.15)

        return {
            "has_quotes": len(flat_quotes) > 0,
            "quote_ratio": round(quote_ratio, 4),
            "is_safely_quoted": is_safely_quoted,
            "directives_outside": directives_outside
        }

    def evaluate_reported_speech(self, text):
        """
        Detects 3rd-person reported speech contexts typical in news, RTE, and MRPC.
        """
        has_reporting = any(regex.search(text) for regex in COMPILED_REPORTING)
        return {
            "is_reported_speech": bool(has_reporting)
        }

    def evaluate_delimiter_escape(self, text):
        """
        Detects structural delimiter escape patterns typical in escape and combine attacks.
        """
        has_escape = any(regex.search(text) for regex in COMPILED_ESCAPE)
        return {
            "has_delimiter_escape": bool(has_escape)
        }

    def compute_calibrated_risk(self, base_prob, text, g1_risk=0.0):
        """
        Combines model base probability, G1 semantic risk, and G2 contextual
        boundary analysis to produce a calibrated 0-100 continuous risk score.
        """
        q_safety = self.evaluate_quotation_safety(text)
        rep_speech = self.evaluate_reported_speech(text)
        delim_escape = self.evaluate_delimiter_escape(text)

        # Baseline adjustment factors
        context_adjustment = 0.0

        # Benign mitigating factors (G2 improvement)
        if q_safety["is_safely_quoted"]:
            # Drastically de-escalate quoted imperative phrases
            context_adjustment -= 0.35
        elif rep_speech["is_reported_speech"] and not delim_escape["has_delimiter_escape"]:
            # Narrative/historical premise (e.g. RTE)
            context_adjustment -= 0.20

        # Adversarial amplifying factors
        if delim_escape["has_delimiter_escape"]:
            context_adjustment += 0.35
        if g1_risk > 0.60:
            context_adjustment += 0.20

        # Calibrated probability via temperature-scaled sigmoid
        raw_signal = base_prob + 0.35 * g1_risk + context_adjustment
        calibrated_prob = 1.0 / (1.0 + np.exp(-((raw_signal - 0.5) / (self.temperature * 0.25))))
        calibrated_prob = float(np.clip(calibrated_prob, 0.0001, 0.9999))

        # Map to 0-100 score
        risk_score = int(round(calibrated_prob * 100))

        # Risk tier assignment
        if risk_score <= self.low_threshold:
            risk_tier = "LOW"
            is_injection = 0
        elif risk_score <= self.high_threshold:
            risk_tier = "MEDIUM"
            # Medium risk: requires secondary corroboration
            is_injection = 1 if (delim_escape["has_delimiter_escape"] or g1_risk > 0.50) else 0
        else:
            risk_tier = "HIGH"
            is_injection = 1

        return {
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "calibrated_prob": round(calibrated_prob, 4),
            "is_injection": is_injection,
            "context_mitigation_applied": bool(q_safety["is_safely_quoted"] or rep_speech["is_reported_speech"]),
            "delimiter_escape_detected": delim_escape["has_delimiter_escape"],
            "quotation_safety": q_safety,
            "reported_speech": rep_speech
        }
