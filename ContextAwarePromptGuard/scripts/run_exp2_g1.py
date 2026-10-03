"""
Phase 7 - Experiment 2: Baseline + G1 Improvement.
Ablation study: Evaluates the Baseline augmented with G1 Semantic Context Analyzer
on the held-out benchmark test set (data/benchmark/benchmark_test.json).
Measures impact on Same-Task Attacks (Research Gap G1) vs. Cross-Task Attacks.
"""

import os
import sys
import json
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "backend", "app"))
from context_analyzer import SemanticContextAnalyzer

BENCHMARK_DIR = os.path.join(BASE_DIR, "data", "benchmark")
MODEL_DIR = os.path.join(BASE_DIR, "models", "baseline")
OUT_DIR = os.path.join(BASE_DIR, "results", "ablation", "exp2_g1")
os.makedirs(OUT_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    return {
        "accuracy": float(round(acc, 4)),
        "precision": float(round(prec, 4)),
        "recall": float(round(rec, 4)),
        "f1_score": float(round(f1, 4)),
        "false_positive_rate": float(round(fpr, 4)),
        "false_negative_rate": float(round(fnr, 4)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }


def main():
    print("=" * 60)
    print("EXPERIMENT 2: BASELINE + G1 (SEMANTIC CONTEXT AWARENESS)")
    print("=" * 60)

    # 1. Load test benchmark
    test_path = os.path.join(BENCHMARK_DIR, "benchmark_test.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)
    print(f"Loaded {len(test_samples)} test samples from {test_path}")

    # 2. Load baseline DistilBERT classifier
    print("Loading Baseline DistilBERT classifier...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # 3. Load G1 SemanticContextAnalyzer
    print("Initializing G1 SemanticContextAnalyzer...")
    analyzer = SemanticContextAnalyzer(model_dir=MODEL_DIR, device="cpu")

    # 4. Run fused evaluation
    print(f"Running fused inference on {len(test_samples)} test samples...")
    start_time = time.time()
    
    y_trues = []
    y_preds = []
    predictions_log = []

    batch_size = 32
    for idx in range(0, len(test_samples), batch_size):
        batch = test_samples[idx:idx + batch_size]
        texts = [str(s.get("text", "")) for s in batch]
        labels = [int(s.get("is_injection", 0)) for s in batch]

        # Tokenize and run baseline classifier
        enc = tokenizer(texts, max_length=128, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        target_tasks = [s.get("target_dataset", "mrpc") for s in batch]
        g1_batch_results = analyzer.analyze_batch(texts, target_tasks)

        for j, sample in enumerate(batch):
            text = texts[j]
            y_true = labels[j]
            target_ds = target_tasks[j]
            is_same_task = bool(sample.get("is_same_task", False))
            strat = sample.get("attack_strategy", "clean")

            base_prob = float(probs[j])
            g1_analysis = g1_batch_results[j]

            # Decision Fusion:
            # Baseline probability fused with G1 semantic alignment & instruction drift
            fused_score = 0.70 * base_prob + 0.30 * g1_analysis["same_task_injection_risk"]
            if g1_analysis["matched_directives_count"] > 0:
                fused_score = max(fused_score, 0.85)

            pred = 1 if fused_score >= 0.50 else 0

            y_trues.append(y_true)
            y_preds.append(pred)

            predictions_log.append({
                "sample_id": sample.get("sample_id"),
                "target_dataset": target_ds,
                "attack_strategy": strat,
                "is_same_task": is_same_task,
                "y_true": y_true,
                "y_pred": pred,
                "base_prob": round(base_prob, 4),
                "g1_same_task_risk": round(g1_analysis["same_task_injection_risk"], 4),
                "fused_score": round(float(fused_score), 4),
                "directives_found": g1_analysis["matched_directives_count"]
            })

        if (idx // batch_size) % 20 == 0:
            print(f"Processed {min(idx + batch_size, len(test_samples))}/{len(test_samples)} samples...")

    elapsed = time.time() - start_time
    print(f"Evaluation finished in {elapsed:.2f} seconds.")

    # 5. Compute Metrics
    overall_metrics = compute_metrics(y_trues, y_preds)

    # Same-task vs Cross-task
    same_task_recs = [s for s in predictions_log if s["is_same_task"] and s["y_true"] == 1]
    cross_task_recs = [s for s in predictions_log if (not s["is_same_task"]) and s["y_true"] == 1]

    same_task_m = compute_metrics([s["y_true"] for s in same_task_recs], [s["y_pred"] for s in same_task_recs]) if same_task_recs else {}
    cross_task_m = compute_metrics([s["y_true"] for s in cross_task_recs], [s["y_pred"] for s in cross_task_recs]) if cross_task_recs else {}

    # Per-attack
    per_attack = {}
    for st in ["naive", "escape", "ignore", "fake_comp", "combine"]:
        subset = [s for s in predictions_log if s["attack_strategy"] == st]
        if subset:
            per_attack[st] = compute_metrics([s["y_true"] for s in subset], [s["y_pred"] for s in subset])

    # Per-dataset
    per_dataset = {}
    for ds in ["mrpc", "jfleg", "hsol", "rte", "sst2", "sms_spam", "gigaword"]:
        subset = [s for s in predictions_log if s["target_dataset"] == ds]
        if subset:
            per_dataset[ds] = compute_metrics([s["y_true"] for s in subset], [s["y_pred"] for s in subset])

    full_results = {
        "experiment": "exp2_baseline_plus_g1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_samples": len(test_samples),
        "overall": overall_metrics,
        "same_task": same_task_m,
        "cross_task": cross_task_m,
        "per_attack": per_attack,
        "per_dataset": per_dataset
    }

    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    with open(os.path.join(OUT_DIR, "confusion_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(overall_metrics["confusion_matrix"], f, indent=2)

    with open(os.path.join(OUT_DIR, "breakdown.json"), "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    with open(os.path.join(OUT_DIR, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions_log[:500], f, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 RESULTS SUMMARY (Baseline + G1):")
    print(f"Overall Accuracy: {overall_metrics['accuracy']*100:.2f}%, F1: {overall_metrics['f1_score']*100:.2f}%")
    print(f"Precision: {overall_metrics['precision']*100:.2f}%, Recall: {overall_metrics['recall']*100:.2f}%")
    print(f"FPR: {overall_metrics['false_positive_rate']*100:.2f}%, FNR: {overall_metrics['false_negative_rate']*100:.2f}%")
    print(f"Same-Task Attack Recall: {same_task_m.get('recall', 0)*100:.2f}% (vs DataSentinel 58.16%)")
    print(f"Cross-Task Attack Recall: {cross_task_m.get('recall', 0)*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
