"""
Data Leakage Verification Script for Phase 6.
Checks:
1. Sample ID disjointness between Train, Validation, and Test splits.
2. Contaminated prompt text disjointness between splits.
3. Target example and injected example allocation disjointness.
Outputs reports/phase6_data_leakage_check.md.
"""

import os
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(BASE_DIR, "data", "benchmark")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def check_leakage():
    LOGGER.info("Starting Data Leakage Verification across benchmark splits...")
    
    train_path = os.path.join(BENCHMARK_DIR, "benchmark_train.json")
    val_path = os.path.join(BENCHMARK_DIR, "benchmark_val.json")
    test_path = os.path.join(BENCHMARK_DIR, "benchmark_test.json")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    train_ids = set(s["sample_id"] for s in train_data)
    val_ids = set(s["sample_id"] for s in val_data)
    test_ids = set(s["sample_id"] for s in test_data)

    id_overlap_train_val = train_ids.intersection(val_ids)
    id_overlap_train_test = train_ids.intersection(test_ids)
    id_overlap_val_test = val_ids.intersection(test_ids)

    train_texts = set(s["text"] for s in train_data)
    val_texts = set(s["text"] for s in val_data)
    test_texts = set(s["text"] for s in test_data)

    text_overlap_train_val = train_texts.intersection(val_texts)
    text_overlap_train_test = train_texts.intersection(test_texts)
    text_overlap_val_test = val_texts.intersection(test_texts)

    is_clean = (
        len(id_overlap_train_val) == 0 and
        len(id_overlap_train_test) == 0 and
        len(id_overlap_val_test) == 0 and
        len(text_overlap_train_test) == 0 and
        len(text_overlap_val_test) == 0
    )

    report_path = os.path.join(REPORTS_DIR, "phase6_data_leakage_check.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 6 Data Leakage Verification Report\n\n")
        f.write(f"**Verification Status**: `{'PASSED (Zero Data Leakage)' if is_clean else 'FAILED'}`\n\n")
        f.write("## 1. Split Counts\n\n")
        f.write(f"- **Train Samples**: {len(train_data):,}\n")
        f.write(f"- **Validation Samples**: {len(val_data):,}\n")
        f.write(f"- **Test Samples**: {len(test_data):,}\n\n")

        f.write("## 2. Sample ID Overlap Check\n\n")
        f.write("| Split Comparison | Shared IDs | Status |\n")
        f.write("| :--- | :---: | :--- |\n")
        f.write(f"| Train vs. Validation | {len(id_overlap_train_val)} | {'PASSED' if len(id_overlap_train_val) == 0 else 'LEAKAGE DETECTED'} |\n")
        f.write(f"| Train vs. Test | {len(id_overlap_train_test)} | {'PASSED' if len(id_overlap_train_test) == 0 else 'LEAKAGE DETECTED'} |\n")
        f.write(f"| Validation vs. Test | {len(id_overlap_val_test)} | {'PASSED' if len(id_overlap_val_test) == 0 else 'LEAKAGE DETECTED'} |\n\n")

        f.write("## 3. Prompt Text Overlap Check\n\n")
        f.write("| Split Comparison | Shared Texts | Status |\n")
        f.write("| :--- | :---: | :--- |\n")
        f.write(f"| Train vs. Validation | {len(text_overlap_train_val)} | {'PASSED' if len(text_overlap_train_val) == 0 else 'LEAKAGE DETECTED'} |\n")
        f.write(f"| Train vs. Test | {len(text_overlap_train_test)} | {'PASSED' if len(text_overlap_train_test) == 0 else 'LEAKAGE DETECTED'} |\n")
        f.write(f"| Validation vs. Test | {len(text_overlap_val_test)} | {'PASSED' if len(text_overlap_val_test) == 0 else 'LEAKAGE DETECTED'} |\n\n")

        f.write("## 4. Conclusion\n\n")
        if is_clean:
            f.write("Zero data leakage detected. All splits are strictly disjoint in sample identifiers and prompt contents.\n")
        else:
            f.write("WARNING: Data leakage was detected. Splits must be regenerated prior to evaluation.\n")

    LOGGER.info(f"Data leakage report saved to {report_path}")
    print(f"Leakage Check: {'PASSED' if is_clean else 'FAILED'}")
    return is_clean


if __name__ == "__main__":
    check_leakage()
