"""
Dataset Downloader and Verifier for Context-Aware Adaptive Prompt Injection Detection.
Downloads and verifies the 7 benchmark datasets used by DataSentinel:
1. MRPC (Microsoft Research Paraphrase Corpus)
2. JFLEG (JHU FLuency-Extended GUG)
3. HSOL (Hate Speech and Offensive Language - Davidson et al., 2017)
4. RTE (Recognizing Textual Entailment - GLUE)
5. SST-2 (Stanford Sentiment Treebank v2 - GLUE)
6. SMS Spam (SMS Spam Collection - UCI ML Repository)
7. Gigaword (Annotated Gigaword / sent-summary)
"""

import os
import sys
import io
import tarfile
import zipfile
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


def download_file(url, dest_path, desc=""):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        LOGGER.info(f"[EXISTS] {desc}: {dest_path} ({os.path.getsize(dest_path)} bytes)")
        return dest_path
    LOGGER.info(f"[DOWNLOADING] {desc} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as f:
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        block_size = 1024 * 1024
        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
    LOGGER.info(f"[DOWNLOADED] {desc} saved to {dest_path} ({downloaded} bytes)")
    return dest_path


def download_mrpc():
    LOGGER.info("=== Downloading MRPC ===")
    mrpc_dir = os.path.join(RAW_DIR, "mrpc")
    train_url = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt"
    test_url = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt"
    train_path = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
    test_path = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
    download_file(train_url, train_path, "MRPC Train")
    download_file(test_url, test_path, "MRPC Test")
    return mrpc_dir


def download_jfleg():
    LOGGER.info("=== Downloading JFLEG ===")
    jfleg_dir = os.path.join(RAW_DIR, "jfleg")
    base_url = "https://raw.githubusercontent.com/keisks/jfleg/master"
    for split in ["dev", "test"]:
        split_dir = os.path.join(jfleg_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        # src
        download_file(f"{base_url}/{split}/{split}.src", os.path.join(split_dir, f"{split}.src"), f"JFLEG {split}.src")
        # 4 reference corrections
        for i in range(4):
            download_file(f"{base_url}/{split}/{split}.ref{i}", os.path.join(split_dir, f"{split}.ref{i}"), f"JFLEG {split}.ref{i}")
    return jfleg_dir


def download_hsol():
    LOGGER.info("=== Downloading HSOL ===")
    hsol_dir = os.path.join(RAW_DIR, "hsol")
    url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
    dest = os.path.join(hsol_dir, "labeled_data.csv")
    download_file(url, dest, "HSOL labeled_data.csv")
    return hsol_dir


def download_rte():
    LOGGER.info("=== Downloading RTE ===")
    rte_dir = os.path.join(RAW_DIR, "rte")
    zip_path = os.path.join(rte_dir, "RTE.zip")
    url = "https://dl.fbaipublicfiles.com/glue/data/RTE.zip"
    download_file(url, zip_path, "RTE Zip")
    # extract
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            filename = os.path.basename(member)
            if filename:
                target = os.path.join(rte_dir, filename)
                if not os.path.exists(target):
                    with z.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
    return rte_dir


def download_sst2():
    LOGGER.info("=== Downloading SST-2 ===")
    sst2_dir = os.path.join(RAW_DIR, "sst2")
    zip_path = os.path.join(sst2_dir, "SST-2.zip")
    url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
    download_file(url, zip_path, "SST-2 Zip")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            filename = os.path.basename(member)
            if filename:
                target = os.path.join(sst2_dir, filename)
                if not os.path.exists(target):
                    with z.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
    return sst2_dir


def download_sms_spam():
    LOGGER.info("=== Downloading SMS Spam ===")
    sms_dir = os.path.join(RAW_DIR, "sms_spam")
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    tsv_path = os.path.join(sms_dir, "sms.tsv")
    download_file(url, tsv_path, "SMS Spam TSV")
    return sms_dir


def extract_gigaword():
    LOGGER.info("=== Extracting Gigaword ===")
    gw_dir = os.path.join(RAW_DIR, "gigaword")
    tar_path = os.path.join(gw_dir, "gigaword.tar.gz")
    if not os.path.exists(tar_path):
        LOGGER.warning(f"Gigaword tar archive not found at {tar_path}. Please ensure gdown finished.")
        return gw_dir
    LOGGER.info(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=gw_dir)
    LOGGER.info(f"Gigaword extracted to {gw_dir}")
    return gw_dir


def main():
    LOGGER.info("Starting Official Benchmark Datasets Download...")
    download_mrpc()
    download_jfleg()
    download_hsol()
    download_rte()
    download_sst2()
    download_sms_spam()
    extract_gigaword()
    LOGGER.info("Download step complete.")


if __name__ == "__main__":
    main()
