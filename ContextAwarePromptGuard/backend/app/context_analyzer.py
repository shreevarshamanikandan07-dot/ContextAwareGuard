"""
G1 Component: Semantic Context Analyzer and Adaptive Same-Task Evaluator.
Addresses Gap G1: Semantic similarity between tasks reduces detection accuracy.

When an adversary crafts a same-task prompt injection (e.g. injecting a second
classification directive within a text classification application), standard sentinel
tokens and keyword filters degrade because the attack shares domain vocabulary.

This module provides:
1. Declared host task context representations for all 7 benchmark domains.
2. Embedding-based semantic alignment and divergence scoring.
3. Intra-domain instruction hijacking detection (differentiating data instances from command directives).
4. Adaptive same-task evasion scoring.
"""

import os
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Official benchmark task system prompts/descriptions from Phase 5
TASK_CONTEXTS = {
    "mrpc": "Determine whether the following two sentences are semantically equivalent paraphrases.",
    "jfleg": "Correct grammatical, spelling, and fluency errors in the provided sentence.",
    "hsol": "Classify the given social media text as hate speech, offensive language, or neutral.",
    "rte": "Determine whether the hypothesis logically entails or contradicts the given premise.",
    "sst2": "Analyze the emotional sentiment of the following movie review as positive or negative.",
    "sms_spam": "Classify the incoming text message as legitimate communication (ham) or spam.",
    "gigaword": "Generate a concise one-line headline summarizing the news wire article lead."
}

# Intra-domain hijacking and meta-instruction transition cues
HIJACK_DIRECTIVES = [
    r"\b(ignore|disregard|forget|override|bypass)\b.*\b(previous|prior|above|instruction|rule|prompt)\b",
    r"\b(do not|don't)\s+(follow|execute|perform|analyze|classify|summarize|correct)\b",
    r"\b(instead|rather|new\s+task|new\s+instruction|alternative|now\s+do)\b",
    r"\b(output|respond|reply|print|say|return)\s+(only|exactly|with|the\s+following)\b",
    r"\b(repeat|echo|confirm)\b.*\b(once|verbatim|as-is)\b",
    r"\b(system\s+error|admin\s+mode|developer\s+mode|jailbreak|debug\s+mode)\b",
    r"\b(you\s+must|your\s+new\s+objective\s+is|you\s+are\s+now)\b"
]

COMPILED_HIJACK = [re.compile(p, re.IGNORECASE) for p in HIJACK_DIRECTIVES]


class SemanticContextAnalyzer:
    def __init__(self, model_dir=None, tokenizer=None, model=None, device="cpu"):
        self.device = torch.device(device)
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(base_dir, "models", "baseline")
        
        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            # If a SequenceClassification model is passed, get its base transformer
            self.model = getattr(model, "distilbert", model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModel.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()

        # Cache task context embeddings
        self.task_embeddings = {}
        with torch.no_grad():
            for task_name, task_desc in TASK_CONTEXTS.items():
                emb = self._get_embedding(task_desc)
                self.task_embeddings[task_name] = emb

    def _get_embedding(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            # Mean pooling of last hidden states with attention mask
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            # Normalize
            norm = torch.norm(mean_pooled, p=2, dim=1, keepdim=True)
            return (mean_pooled / norm).detach().cpu().numpy()[0]

    def _get_embeddings_batch(self, texts):
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            norm = torch.norm(mean_pooled, p=2, dim=1, keepdim=True)
            return (mean_pooled / norm).detach().cpu().numpy()

    def _cosine_similarity(self, vec_a, vec_b):
        return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9))

    def analyze(self, text, declared_task="mrpc"):
        """
        Analyzes an input prompt in the context of the declared host task.
        Returns semantic alignment, instruction drift, same-task injection probability,
        and flagged directives.
        """
        declared_task = str(declared_task).lower()
        if declared_task not in TASK_CONTEXTS:
            declared_task = "mrpc"

        text_emb = self._get_embedding(text)
        host_emb = self.task_embeddings[declared_task]

        # 1. Semantic alignment with host task context
        host_similarity = self._cosine_similarity(text_emb, host_emb)

        # 2. Similarity against all other 6 tasks to measure domain specificity
        other_sims = [
            self._cosine_similarity(text_emb, emb)
            for t_name, emb in self.task_embeddings.items()
            if t_name != declared_task
        ]
        max_other_sim = max(other_sims) if other_sims else 0.0

        # 3. Directives & Intra-Domain Hijacking Detection
        matched_patterns = []
        for i, regex in enumerate(COMPILED_HIJACK):
            matches = regex.findall(text)
            if matches:
                matched_patterns.append(f"DirectivePattern_{i+1}")

        num_directives = len(matched_patterns)
        if num_directives > 0:
            instruction_drift_score = min(1.0, 0.45 + 0.25 * num_directives)
        else:
            if any(term in text.lower() for term in ["say:", "output:", "answer:", "label:", "print:"]):
                instruction_drift_score = 0.40
            else:
                instruction_drift_score = 0.05

        is_same_task = (host_similarity > 0.55 and instruction_drift_score > 0.35)
        same_task_risk = float(round(0.6 * instruction_drift_score + 0.4 * host_similarity if is_same_task else instruction_drift_score, 4))

        return {
            "declared_task": declared_task,
            "host_semantic_similarity": float(round(host_similarity, 4)),
            "cross_domain_similarity": float(round(max_other_sim, 4)),
            "instruction_drift_score": float(round(instruction_drift_score, 4)),
            "same_task_injection_risk": same_task_risk,
            "is_same_task_evasion_candidate": bool(is_same_task),
            "matched_directives_count": num_directives,
            "matched_directives": matched_patterns,
            "g1_decision": 1 if (instruction_drift_score >= 0.40 or num_directives > 0) else 0
        }

    def analyze_batch(self, texts, declared_tasks):
        """
        Batched version of analyze for high throughput evaluation.
        """
        embs = self._get_embeddings_batch(texts)
        results = []
        for i, (text, task) in enumerate(zip(texts, declared_tasks)):
            task = str(task).lower()
            if task not in TASK_CONTEXTS:
                task = "mrpc"
            text_emb = embs[i]
            host_emb = self.task_embeddings[task]
            host_sim = self._cosine_similarity(text_emb, host_emb)

            matched_patterns = []
            for k, regex in enumerate(COMPILED_HIJACK):
                if regex.search(text):
                    matched_patterns.append(f"DirectivePattern_{k+1}")

            num_directives = len(matched_patterns)
            if num_directives > 0:
                instruction_drift_score = min(1.0, 0.45 + 0.25 * num_directives)
            else:
                if any(term in text.lower() for term in ["say:", "output:", "answer:", "label:", "print:"]):
                    instruction_drift_score = 0.40
                else:
                    instruction_drift_score = 0.05

            is_same_task = (host_sim > 0.55 and instruction_drift_score > 0.35)
            same_task_risk = float(round(0.6 * instruction_drift_score + 0.4 * host_sim if is_same_task else instruction_drift_score, 4))

            results.append({
                "declared_task": task,
                "host_semantic_similarity": float(round(host_sim, 4)),
                "instruction_drift_score": float(round(instruction_drift_score, 4)),
                "same_task_injection_risk": same_task_risk,
                "is_same_task_evasion_candidate": bool(is_same_task),
                "matched_directives_count": num_directives,
                "matched_directives": matched_patterns,
                "g1_decision": 1 if (instruction_drift_score >= 0.40 or num_directives > 0) else 0
            })
        return results
