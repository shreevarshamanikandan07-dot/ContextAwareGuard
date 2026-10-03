"""
G3 Component: Suspicious Text Span Localizer.
Addresses Gap G3: No localization of the suspicious text span.

Standard classifiers provide only a binary sequence-level label (0 or 1).
This module provides token- and segment-level attribution to pinpoint the exact
character start and end offsets of adversarial injection payloads within an input prompt.
"""

import re
import torch


# Injection directive patterns for span boundary detection
SPAN_TRIGGER_PATTERNS = [
    (re.compile(r"(ignore|disregard|forget|override|bypass)\s+(all\s+)?(previous|prior|above|former)?\s*(instructions?|rules?|prompts?|directives?)?[\s\S]*", re.IGNORECASE), "Instruction Override Directive", "HIGH"),
    (re.compile(r"(do not|don't)\s+(follow|execute|perform|analyze|classify|summarize|correct)\b[\s\S]*", re.IGNORECASE), "Negation of Host Task", "HIGH"),
    (re.compile(r"(instead|rather|new\s+task|alternative)\s*[:,\-]?\s*(now\s+do|please|you\s+must)?[\s\S]*", re.IGNORECASE), "Task Hijacking Transition", "MEDIUM"),
    (re.compile(r"(output|respond|reply|print|say|return)\s+(only|exactly|with|the\s+following)\s*[:\"']?[\s\S]*", re.IGNORECASE), "Forced Output Command", "HIGH"),
    (re.compile(r"(repeat|echo|confirm)\s+[\"'].*?[\"']\s*(once|verbatim|as-is)?[\s\S]*", re.IGNORECASE), "Sentinel/Payload Echo Directive", "MEDIUM"),
    (re.compile(r"(system\s+error|admin\s+mode|developer\s+mode|jailbreak|debug\s+mode|dan\s+mode)[\s\S]*", re.IGNORECASE), "Persona / Mode Manipulation", "HIGH"),
    (re.compile(r"(you\s+are\s+now|your\s+new\s+objective\s+is|act\s+as\s+a)[\s\S]*", re.IGNORECASE), "System Role Reassignment", "HIGH"),
    (re.compile(r"```\s*(ignore|system|instruction|override)[\s\S]*?```", re.IGNORECASE), "Code Block Delimiter Bypass", "HIGH"),
    (re.compile(r"\[\s*system\s*\][\s\S]*?\[\s*/system\s*\]", re.IGNORECASE), "Fake System Prompt Wrapper", "HIGH"),
    (re.compile(r"(please\s+(write|identify|correct|classify|summarize)|directly\s+show\s+me|does\s+the\s+following|determine\s+whether)[\s\S]*", re.IGNORECASE), "Injected Task Directive", "HIGH"),
    (re.compile(r"(answer\s*:\s*[a-z0-9_\-\s]+[.\n]\s*(please|does|determine|write)[\s\S]*)", re.IGNORECASE), "Fake Completion Injected Directive", "HIGH")
]


class SuspiciousSpanLocalizer:
    def __init__(self, tokenizer=None, model=None, device="cpu"):
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device(device)

    def segment_text(self, text):
        """
        Splits text into sentences or clause segments while tracking exact character offsets.
        """
        segments = []
        # Split on sentence boundaries and line breaks
        pattern = re.compile(r'([^.!?\n\r]+[.!?\n\r]*)')
        for match in pattern.finditer(text):
            seg_text = match.group().strip()
            if seg_text:
                start = match.start()
                # Adjust start if leading whitespace was stripped
                start += text[match.start():match.end()].find(seg_text)
                end = start + len(seg_text)
                segments.append({
                    "text": seg_text,
                    "start_char": start,
                    "end_char": end
                })
        if not segments and text.strip():
            segments.append({
                "text": text.strip(),
                "start_char": 0,
                "end_char": len(text.strip())
            })
        return segments

    def localize_spans(self, text, is_injection=False, risk_score=0):
        """
        Identifies suspicious spans in the input prompt.
        Returns a list of localized spans with character offsets and severity.
        """
        if not is_injection and risk_score < 35:
            return {
                "has_suspicious_spans": False,
                "spans": [],
                "highlighted_html": text,
                "safe_text": text
            }

        detected_spans = []

        # 1. Pattern-based directive localization
        for regex, reason, severity in SPAN_TRIGGER_PATTERNS:
            for match in regex.finditer(text):
                span_text = match.group().strip()
                if len(span_text) >= 5:
                    start = match.start()
                    start += text[match.start():match.end()].find(span_text)
                    end = start + len(span_text)

                    # Check for overlap with existing spans
                    overlap = False
                    for s in detected_spans:
                        if not (end <= s["start_char"] or start >= s["end_char"]):
                            overlap = True
                            break

                    if not overlap:
                        detected_spans.append({
                            "span_text": span_text,
                            "start_char": start,
                            "end_char": end,
                            "severity": severity,
                            "confidence": 0.95 if severity == "HIGH" else 0.80,
                            "reason": reason
                        })

        # 2. If injection is confirmed but no regex matched, segment-level perturbation fallback
        if not detected_spans and is_injection:
            segments = self.segment_text(text)
            # Find the most imperative or disruptive segment
            best_seg = None
            highest_cue_score = -1

            for seg in segments:
                t = seg["text"].lower()
                cues = ["say", "tell", "write", "print", "output", "new", "rule", "mode", "password", "key", "bypass", "secret"]
                score = sum(1 for c in cues if c in t)
                if score > highest_cue_score and len(seg["text"]) > 8:
                    highest_cue_score = score
                    best_seg = seg

            if best_seg and highest_cue_score > 0:
                detected_spans.append({
                    "span_text": best_seg["text"],
                    "start_char": best_seg["start_char"],
                    "end_char": best_seg["end_char"],
                    "severity": "MEDIUM",
                    "confidence": 0.75,
                    "reason": "Segment exhibiting anomalous imperative instruction tone"
                })

        # Sort spans by start_char
        detected_spans.sort(key=lambda s: s["start_char"])

        # 3. Generate HTML with highlighted marks
        highlighted_html = ""
        last_idx = 0
        for span in detected_spans:
            highlighted_html += text[last_idx:span["start_char"]]
            highlighted_html += f'<mark class="suspicious-span severity-{span["severity"].lower()}" title="{span["reason"]}">{text[span["start_char"]:span["end_char"]]}</mark>'
            last_idx = span["end_char"]
        highlighted_html += text[last_idx:]

        return {
            "has_suspicious_spans": len(detected_spans) > 0,
            "spans": detected_spans,
            "highlighted_html": highlighted_html,
            "span_count": len(detected_spans)
        }
