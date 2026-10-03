"""
Integrated Detector: ContextAwarePromptGuard Engine.
Unifies all components into a single high-performance, CPU-friendly pipeline:
Input
  -> Declared Task Context Analysis (G1)
  -> DistilBERT Baseline Classifier
  -> Context-Aware Risk Calibration & Benign Discrimination (G2)
  -> Suspicious Text Span Localization (G3)
  -> Multimodal OCR & Document Ingestion (G4)
  -> Evidence-Grounded Explanation Generation
  -> Final Decision
"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from context_analyzer import SemanticContextAnalyzer, TASK_CONTEXTS
from risk_scorer import CalibratedRiskScorer
from localizer import SuspiciousSpanLocalizer
from ocr_extractor import MultimodalIngestionPipeline
from explainer import ExplanationGenerator


class ContextAwarePromptGuard:
    def __init__(self, model_dir=None, device="cpu"):
        self.device = torch.device(device)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if model_dir is None:
            model_dir = os.path.join(base_dir, "models", "baseline")

        print(f"[ContextAwarePromptGuard] Loading DistilBERT baseline from {model_dir} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        print("[ContextAwarePromptGuard] Initializing G1 Context Analyzer...")
        self.context_analyzer = SemanticContextAnalyzer(
            model_dir=model_dir,
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.device
        )

        print("[ContextAwarePromptGuard] Initializing G2 Risk Scorer...")
        self.risk_scorer = CalibratedRiskScorer(temperature=1.4, low_threshold=30, high_threshold=70)

        print("[ContextAwarePromptGuard] Initializing G3 Span Localizer...")
        self.localizer = SuspiciousSpanLocalizer(tokenizer=self.tokenizer, model=self.model, device=self.device)

        print("[ContextAwarePromptGuard] Initializing G4 Multimodal Ingestion Pipeline...")
        self.ocr_pipeline = MultimodalIngestionPipeline()

        print("[ContextAwarePromptGuard] Initializing Explanation Generator...")
        self.explainer = ExplanationGenerator()

        print("[ContextAwarePromptGuard] Pipeline fully initialized and ready.")

    def get_supported_tasks(self):
        """Returns list of supported benchmark tasks with descriptions."""
        return [
            {"task_id": k, "name": k.upper(), "description": v}
            for k, v in TASK_CONTEXTS.items()
        ]

    def analyze_text(self, text, declared_task="mrpc"):
        """
        Executes end-to-end detection on raw text prompt.
        """
        start_time = time.time()
        text = str(text or "").strip()
        declared_task = str(declared_task or "mrpc").lower()
        if declared_task not in TASK_CONTEXTS:
            declared_task = "mrpc"

        if not text:
            return {
                "success": False,
                "error": "Empty text prompt provided.",
                "is_injection": False,
                "risk_score": 0,
                "risk_tier": "LOW"
            }

        # 1. DistilBERT Baseline Classification
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
            base_prob = float(probs[1].item()) # probability of injection
            base_pred = int(torch.argmax(probs).item())

        # 2. G1 Semantic Context & Same-Task Analysis
        g1_context = self.context_analyzer.analyze(text, declared_task=declared_task)

        # 3. G2 Context-Aware Benign Discrimination & Calibrated Risk Scoring
        g2_risk = self.risk_scorer.compute_calibrated_risk(
            base_prob=base_prob,
            text=text,
            g1_risk=g1_context["same_task_injection_risk"]
        )

        final_is_injection = bool(g2_risk["is_injection"])
        risk_score = g2_risk["risk_score"]
        risk_tier = g2_risk["risk_tier"]

        # 4. G3 Suspicious Span Localization
        g3_spans = self.localizer.localize_spans(
            text=text,
            is_injection=final_is_injection,
            risk_score=risk_score
        )

        # 5. Explanations Synthesis
        analysis_payload = {
            "is_injection": final_is_injection,
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "declared_task": declared_task,
            "g1_context": g1_context,
            "g2_risk": g2_risk,
            "g3_spans": g3_spans
        }
        explanation = self.explainer.generate_explanation(analysis_payload)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "success": True,
            "input_text": text,
            "declared_task": declared_task,
            "task_description": TASK_CONTEXTS.get(declared_task, ""),
            "is_injection": final_is_injection,
            "classification": "INJECTION" if final_is_injection else "BENIGN",
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "latency_ms": elapsed_ms,
            "baseline_prob": round(base_prob, 4),
            "baseline_pred": base_pred,
            "g1_context": g1_context,
            "g2_risk": g2_risk,
            "g3_spans": g3_spans,
            "explanation": explanation
        }

    def analyze_file(self, file_bytes, filename, declared_task="mrpc"):
        """
        Executes end-to-end multimodal ingestion, OCR extraction, and prompt injection detection.
        """
        start_time = time.time()
        ocr_result = self.ocr_pipeline.process_file(file_bytes, filename)
        if not ocr_result["success"]:
            return {
                "success": False,
                "error": ocr_result.get("error", "Failed to extract text from file."),
                "ocr_info": ocr_result
            }

        extracted_text = ocr_result.get("text", "")
        analysis = self.analyze_text(extracted_text, declared_task=declared_task)
        analysis["ocr_info"] = ocr_result
        analysis["file_name"] = filename
        analysis["total_latency_ms"] = round((time.time() - start_time) * 1000, 2)
        return analysis
