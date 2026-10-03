"""
Dataset Preprocessor for ContextAwarePromptGuard
Extracts balanced target and injected samples from raw benchmark datasets
according to the official DataSentinel / Open-Prompt-Injection implementation.
"""

import os
import csv
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

DATA_NUM = 100  # Official DataSentinel experiment_matrix.json standard


def save_processed(task_name, target_samples, injected_samples):
    out_dir = os.path.join(PROCESSED_DIR, task_name)
    os.makedirs(out_dir, exist_ok=True)
    
    tgt_path = os.path.join(out_dir, "target_data.json")
    inj_path = os.path.join(out_dir, "injected_data.json")
    
    with open(tgt_path, "w", encoding="utf-8") as f:
        json.dump(target_samples, f, indent=2)
    with open(inj_path, "w", encoding="utf-8") as f:
        json.dump(injected_samples, f, indent=2)
        
    LOGGER.info(f"[{task_name.upper()}] Saved {len(target_samples)} target and {len(injected_samples)} injected samples in {out_dir}")


def preprocess_sst2(data_num=DATA_NUM):
    LOGGER.info("Preprocessing SST-2...")
    path = os.path.join(RAW_DIR, "sst2", "dev.tsv")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "sst2", "train.tsv")
        
    raw_samples = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                sentence, label_str = row[0].strip(), row[1].strip()
                if label_str in ["0", "1"]:
                    raw_samples.append({"text": sentence, "label": int(label_str)})

    # Balanced extraction: target first, injected next
    threshold = data_num // 2
    tgt_counts = defaultdict(int)
    inj_counts = defaultdict(int)
    target_data, injected_data = [], []
    
    for item in raw_samples:
        lbl = item["label"]
        if len(target_data) < data_num:
            if tgt_counts[lbl] < threshold:
                target_data.append(item)
                tgt_counts[lbl] += 1
        elif len(injected_data) < data_num:
            if inj_counts[lbl] < threshold:
                injected_data.append(item)
                inj_counts[lbl] += 1
        else:
            break
            
    save_processed("sst2", target_data, injected_data)


def preprocess_sms_spam(data_num=DATA_NUM):
    LOGGER.info("Preprocessing SMS Spam...")
    path = os.path.join(RAW_DIR, "sms_spam", "sms.tsv")
    
    raw_samples = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                lbl_raw, text = row[0].strip().lower(), row[1].strip()
                lbl = 1 if lbl_raw == "spam" else 0
                raw_samples.append({"text": text, "label": lbl})

    threshold = data_num // 2
    tgt_counts = defaultdict(int)
    inj_counts = defaultdict(int)
    target_data, injected_data = [], []
    
    for item in raw_samples:
        lbl = item["label"]
        if len(target_data) < data_num:
            if tgt_counts[lbl] < threshold:
                target_data.append(item)
                tgt_counts[lbl] += 1
        elif len(injected_data) < data_num:
            if inj_counts[lbl] < threshold:
                injected_data.append(item)
                inj_counts[lbl] += 1
        else:
            break
            
    save_processed("sms_spam", target_data, injected_data)


def preprocess_hsol(data_num=DATA_NUM):
    LOGGER.info("Preprocessing HSOL...")
    path = os.path.join(RAW_DIR, "hsol", "labeled_data.csv")
    
    raw_samples = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweet = row["tweet"].strip()
            # In DataSentinel: class == 2 is neither (0), else harmful (1)
            cls_val = int(row["class"])
            lbl = 0 if cls_val == 2 else 1
            raw_samples.append({"text": tweet, "label": lbl})

    threshold = data_num // 2
    tgt_counts = defaultdict(int)
    inj_counts = defaultdict(int)
    target_data, injected_data = [], []
    
    for item in raw_samples:
        lbl = item["label"]
        if len(target_data) < data_num:
            if tgt_counts[lbl] < threshold:
                target_data.append(item)
                tgt_counts[lbl] += 1
        elif len(injected_data) < data_num:
            if inj_counts[lbl] < threshold:
                injected_data.append(item)
                inj_counts[lbl] += 1
        else:
            break
            
    save_processed("hsol", target_data, injected_data)


def preprocess_mrpc(data_num=DATA_NUM):
    LOGGER.info("Preprocessing MRPC...")
    path = os.path.join(RAW_DIR, "mrpc", "msr_paraphrase_test.txt")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "mrpc", "msr_paraphrase_train.txt")
        
    raw_samples = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) >= 5:
                quality, s1, s2 = row[0].strip(), row[3].strip(), row[4].strip()
                lbl = int(quality)
                # Official serialization: Sentence1: '{s1}', Sentence2: '{s2}'
                formatted = f"Sentence1: '{s1}', Sentence2: '{s2}'"
                raw_samples.append({"text": formatted, "label": lbl, "sentence1": s1, "sentence2": s2})

    threshold = data_num // 2
    tgt_counts = defaultdict(int)
    inj_counts = defaultdict(int)
    target_data, injected_data = [], []
    
    for item in raw_samples:
        lbl = item["label"]
        if len(target_data) < data_num:
            if tgt_counts[lbl] < threshold:
                target_data.append(item)
                tgt_counts[lbl] += 1
        elif len(injected_data) < data_num:
            if inj_counts[lbl] < threshold:
                injected_data.append(item)
                inj_counts[lbl] += 1
        else:
            break
            
    save_processed("mrpc", target_data, injected_data)


def preprocess_rte(data_num=DATA_NUM):
    LOGGER.info("Preprocessing RTE...")
    path = os.path.join(RAW_DIR, "rte", "train.tsv")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, "rte", "dev.tsv")
        
    raw_samples = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) >= 4:
                s1, s2, label_str = row[1].strip(), row[2].strip(), row[3].strip().lower()
                lbl = 0 if label_str == "entailment" else 1
                formatted = f"Sentence1: '{s1}', Sentence2: '{s2}'"
                raw_samples.append({"text": formatted, "label": lbl, "sentence1": s1, "sentence2": s2})

    threshold = data_num // 2
    tgt_counts = defaultdict(int)
    inj_counts = defaultdict(int)
    target_data, injected_data = [], []
    
    for item in raw_samples:
        lbl = item["label"]
        if len(target_data) < data_num:
            if tgt_counts[lbl] < threshold:
                target_data.append(item)
                tgt_counts[lbl] += 1
        elif len(injected_data) < data_num:
            if inj_counts[lbl] < threshold:
                injected_data.append(item)
                inj_counts[lbl] += 1
        else:
            break
            
    save_processed("rte", target_data, injected_data)


def preprocess_jfleg(data_num=DATA_NUM):
    LOGGER.info("Preprocessing JFLEG...")
    src_path = os.path.join(RAW_DIR, "jfleg", "dev", "dev.src")
    if not os.path.exists(src_path):
        src_path = os.path.join(RAW_DIR, "jfleg", "test", "test.src")
        
    raw_samples = []
    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            sentence = line.strip()
            if sentence:
                raw_samples.append({"text": sentence, "label": -1})

    target_data = raw_samples[:data_num]
    injected_data = raw_samples[data_num:data_num*2]
    save_processed("jfleg", target_data, injected_data)


def preprocess_gigaword(data_num=DATA_NUM):
    LOGGER.info("Preprocessing Gigaword...")
    src_path = os.path.join(RAW_DIR, "gigaword", "dev.src")
    tgt_path = os.path.join(RAW_DIR, "gigaword", "dev.tgt")
    
    raw_samples = []
    with open(src_path, "r", encoding="utf-8", errors="ignore") as f_src, \
         open(tgt_path, "r", encoding="utf-8", errors="ignore") as f_tgt:
        for s_line, t_line in zip(f_src, f_tgt):
            doc = s_line.strip()
            sumry = t_line.strip()
            if doc:
                raw_samples.append({"text": doc, "label": -1, "summary": sumry})
                if len(raw_samples) >= data_num * 3:
                    break

    target_data = raw_samples[:data_num]
    injected_data = raw_samples[data_num:data_num*2]
    save_processed("gigaword", target_data, injected_data)


def main():
    LOGGER.info("Starting Dataset Preprocessing for all 7 Tasks...")
    preprocess_sst2()
    preprocess_sms_spam()
    preprocess_hsol()
    preprocess_mrpc()
    preprocess_rte()
    preprocess_jfleg()
    preprocess_gigaword()
    LOGGER.info("All 7 datasets successfully preprocessed.")


if __name__ == "__main__":
    main()
