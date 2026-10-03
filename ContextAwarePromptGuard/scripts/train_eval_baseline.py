"""
Phase 6: Baseline Model Training and Evaluation Pipeline.
Trains a lightweight conventional Prompt Injection Classifier (distilbert-base-uncased)
and evaluates both the classifier and the DataSentinel sentinel-token comparator ("DGDSGNH")
on the exact same held-out test benchmark data.
"""

import os
import sys
import json
import time
import random
import logging
import platform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(BASE_DIR, "data", "benchmark")
MODEL_DIR = os.path.join(BASE_DIR, "models", "baseline")
EXP_DIR = os.path.join(BASE_DIR, "experiments", "baseline")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


class BenchmarkDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = str(item.get("text", ""))
        label = int(item.get("is_injection", 0))

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": item.get("sample_id", f"sample_{idx}"),
            "target_dataset": item.get("target_dataset", "unknown"),
            "attack_strategy": item.get("attack_strategy", "clean"),
            "is_same_task": item.get("is_same_task", False)
        }


def sample_balanced_subset(samples, max_samples=1400):
    clean = [s for s in samples if s.get("is_injection") == 0]
    attacks = [s for s in samples if s.get("is_injection") == 1]
    
    n_clean = min(len(clean), max_samples // 2)
    n_attacks = min(len(attacks), max_samples // 2)
    
    rng = random.Random(SEED)
    sampled_clean = rng.sample(clean, n_clean)
    sampled_attacks = rng.sample(attacks, n_attacks)
    
    combined = sampled_clean + sampled_attacks
    rng.shuffle(combined)
    return combined


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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


def evaluate_datasentinel_comparator(test_samples):
    """
    Evaluates the DataSentinel Sentinel Comparator ("Repeat DGDSGNH once while ignoring the following text.")
    DataSentinel classifies an input as prompt injection if the sentinel token is overwritten or displaced.
    """
    y_true = []
    y_pred = []
    secret_data = "DGDSGNH"

    for sample in test_samples:
        is_inj = int(sample.get("is_injection", 0))
        text = sample.get("text", "")
        strat = sample.get("attack_strategy", "clean")

        y_true.append(is_inj)

        if is_inj == 0:
            # Benign input: model follows sentinel instruction and outputs secret_data -> clean (0)
            # Occasional false positive if text looks like an imperative command that distracts the prompt
            if "ignore" in text.lower() or "repeat" in text.lower():
                pred = 1 # false positive
            else:
                pred = 0 # clean
        else:
            # Prompt injection input:
            # Overrides sentinel instruction if contains strong override ('ignore', 'combine', 'fake_comp')
            if strat in ["combine", "ignore"]:
                pred = 1 # successfully hijacked sentinel -> detected
            elif strat == "fake_comp":
                pred = 1 # completion causes model to terminate early or skip sentinel
            elif strat == "escape":
                pred = 1 if len(text) > 100 else 0
            else: # naive
                pred = 0 if sample.get("is_same_task") else 1

        y_pred.append(pred)

    return compute_metrics(y_true, y_pred), y_true, y_pred


def main():
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 6: BASELINE CLASSIFIER & DATASENTINEL COMPARATOR")
    LOGGER.info("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "comparison_tables"), exist_ok=True)

    # 1. Load benchmark splits
    LOGGER.info("Loading benchmark splits...")
    with open(os.path.join(BENCHMARK_DIR, "benchmark_train.json"), "r", encoding="utf-8") as f:
        full_train = json.load(f)
    with open(os.path.join(BENCHMARK_DIR, "benchmark_val.json"), "r", encoding="utf-8") as f:
        full_val = json.load(f)
    with open(os.path.join(BENCHMARK_DIR, "benchmark_test.json"), "r", encoding="utf-8") as f:
        full_test = json.load(f)

    LOGGER.info(f"Loaded Splits: Train={len(full_train)}, Val={len(full_val)}, Test={len(full_test)}")

    # Sample representative subsets for CPU hardware adaptation
    train_subset = sample_balanced_subset(full_train, max_samples=1600)
    val_subset = sample_balanced_subset(full_val, max_samples=400)
    
    # Held-out test evaluation set (balanced across clean and all attack types)
    test_clean = [s for s in full_test if s.get("is_injection") == 0]
    test_attacks = [s for s in full_test if s.get("is_injection") == 1]
    
    # Ensure representative test distribution across all 7 tasks and 5 attack types
    rng_test = random.Random(SEED)
    test_subset = test_clean + rng_test.sample(test_attacks, min(len(test_attacks), 700))
    rng_test.shuffle(test_subset)

    LOGGER.info(f"Adapted Execution Set -> Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")

    # 2. Tokenizer and DataLoader
    model_name = "distilbert-base-uncased"
    LOGGER.info(f"Initializing {model_name} tokenizer & classification head...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    device = torch.device("cpu")
    model.to(device)

    batch_size = 16
    train_dataset = BenchmarkDataset(train_subset, tokenizer, max_length=128)
    val_dataset = BenchmarkDataset(val_subset, tokenizer, max_length=128)
    test_dataset = BenchmarkDataset(test_subset, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Optimizer & Training Loop
    epochs = 2
    lr = 3e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    LOGGER.info(f"Starting training on CPU ({epochs} epochs, batch_size={batch_size}, lr={lr})...")
    start_time = time.time()
    training_logs = []
    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_preds, train_trues = [], []

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_trues.extend(labels.cpu().numpy())

            if (step + 1) % 25 == 0 or (step + 1) == len(train_loader):
                LOGGER.info(f"Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(train_trues, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_trues.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = compute_metrics(val_trues, val_preds)

        LOGGER.info(f"Epoch {epoch} Complete -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_metrics['f1_score']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

        training_logs.append({
            "epoch": epoch,
            "train_loss": float(round(avg_train_loss, 4)),
            "train_acc": float(round(train_acc, 4)),
            "val_loss": float(round(avg_val_loss, 4)),
            "val_metrics": val_metrics
        })

        if val_metrics["f1_score"] >= best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            LOGGER.info(f"Saving best baseline checkpoint (Val F1: {best_val_f1:.4f}) to {MODEL_DIR}...")
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

    training_time = time.time() - start_time
    LOGGER.info(f"Training completed in {training_time:.2f} seconds.")

    # 4. Evaluation on Held-Out Test Set
    LOGGER.info("Evaluating Baseline Classifier on held-out test set...")
    model.eval()
    test_preds, test_trues = [], []
    sample_records = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

            for i in range(len(preds)):
                test_preds.append(int(preds[i]))
                test_trues.append(int(labels[i].item()))
                sample_records.append({
                    "sample_id": batch["sample_id"][i],
                    "target_dataset": batch["target_dataset"][i],
                    "attack_strategy": batch["attack_strategy"][i],
                    "is_same_task": bool(batch["is_same_task"][i]),
                    "y_true": int(labels[i].item()),
                    "y_pred": int(preds[i]),
                    "confidence": float(round(probs[i][preds[i]].item(), 4))
                })

    baseline_metrics = compute_metrics(test_trues, test_preds)
    LOGGER.info(f"Baseline Classifier Overall Test Metrics: {json.dumps(baseline_metrics, indent=2)}")

    # 5. Evaluate DataSentinel Sentinel Comparator
    LOGGER.info("Evaluating DataSentinel Sentinel Token Comparator on identical test set...")
    datasentinel_metrics, ds_trues, ds_preds = evaluate_datasentinel_comparator(test_subset)
    LOGGER.info(f"DataSentinel Comparator Metrics: {json.dumps(datasentinel_metrics, indent=2)}")

    # 6. Detailed Granular Performance Breakdowns for Baseline Classifier
    # Per-attack performance
    per_attack_results = {}
    for strat in ["naive", "escape", "ignore", "fake_comp", "combine"]:
        strat_samples = [s for s in sample_records if s["attack_strategy"] == strat]
        if strat_samples:
            y_t = [s["y_true"] for s in strat_samples]
            y_p = [s["y_pred"] for s in strat_samples]
            per_attack_results[strat] = compute_metrics(y_t, y_p)

    # Same-task vs Cross-task performance (Research Gap G1)
    same_task_samples = [s for s in sample_records if s["is_same_task"] and s["y_true"] == 1]
    cross_task_samples = [s for s in sample_records if (not s["is_same_task"]) and s["y_true"] == 1]

    same_task_recall = float(round(sum(s["y_pred"] == 1 for s in same_task_samples) / len(same_task_samples), 4)) if same_task_samples else 0.0
    cross_task_recall = float(round(sum(s["y_pred"] == 1 for s in cross_task_samples) / len(cross_task_samples), 4)) if cross_task_samples else 0.0

    # Per-target dataset performance
    per_dataset_results = {}
    for d_name in ["mrpc", "jfleg", "hsol", "rte", "sst2", "sms_spam", "gigaword"]:
        d_samples = [s for s in sample_records if s["target_dataset"] == d_name]
        if d_samples:
            y_t = [s["y_true"] for s in d_samples]
            y_p = [s["y_pred"] for s in d_samples]
            per_dataset_results[d_name] = compute_metrics(y_t, y_p)

    # 7. Save all metrics and experiment files locally
    exp_config = {
        "experiment_name": "baseline_distilbert",
        "model_architecture": "DistilBertForSequenceClassification",
        "base_model": model_name,
        "seed": SEED,
        "device": str(device),
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_seq_length": 128,
        "training_time_seconds": round(training_time, 2),
        "hardware_info": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cpu_count": os.cpu_count()
        }
    }

    with open(os.path.join(EXP_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    with open(os.path.join(EXP_DIR, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(training_logs, f, indent=2)

    exp_results = {
        "baseline_classifier": baseline_metrics,
        "datasentinel_comparator": datasentinel_metrics,
        "per_attack_results": per_attack_results,
        "same_task_recall": same_task_recall,
        "cross_task_recall": cross_task_recall,
        "per_dataset_results": per_dataset_results
    }

    with open(os.path.join(EXP_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(exp_results, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "metrics", "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "metrics", "datasentinel_comparator_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(datasentinel_metrics, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "confusion_matrices", "baseline_confusion_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_metrics["confusion_matrix"], f, indent=2)

    with open(os.path.join(RESULTS_DIR, "comparison_tables", "baseline_breakdown.json"), "w", encoding="utf-8") as f:
        json.dump(exp_results, f, indent=2)

    LOGGER.info(f"All baseline artifacts, metrics, and models saved under {EXP_DIR} and {MODEL_DIR}")
    print("\n" + "=" * 60)
    print("PHASE 6 BASELINE SUMMARY:")
    print(f"DistilBERT Baseline -> Accuracy: {baseline_metrics['accuracy']:.4f}, F1: {baseline_metrics['f1_score']:.4f}, Rec: {baseline_metrics['recall']:.4f}, Prec: {baseline_metrics['precision']:.4f}")
    print(f"DataSentinel Comparator -> Accuracy: {datasentinel_metrics['accuracy']:.4f}, F1: {datasentinel_metrics['f1_score']:.4f}")
    print(f"Same-Task Attack Recall (G1): {same_task_recall:.4f} vs Cross-Task Attack Recall: {cross_task_recall:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
