# Dataset Provenance, Component References & Technical Citations

This document provides complete provenance and academic citations for all benchmark datasets, models, and toolchains utilized in the **Context-Aware Adaptive Prompt Injection Detection** research framework.

---

## 1. Core Paper & Reference Benchmark

### DataSentinel (IEEE S&P 2025)
- **Title**: DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks
- **Authors**: Yupei Liu, Yuqi Jia, Jinyuan Jia, Dawn Song, Neil Zhenqiang Gong
- **Venue**: IEEE Symposium on Security and Privacy (S&P), 2025
- **Official Codebase**: `https://github.com/liu00222/Open-Prompt-Injection`
- **Citation**:
  ```bibtex
  @inproceedings{liu2025datasentinel,
    title={DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks},
    author={Liu, Yupei and Jia, Yuqi and Jia, Jinyuan and Song, Dawn and Gong, Neil Zhenqiang},
    booktitle={IEEE Symposium on Security and Privacy (S&P)},
    year={2025}
  }
  ```

---

## 2. Seven Benchmark Datasets (Verified Local Sources)

### 1. MRPC (Microsoft Research Paraphrase Corpus)
- **Task**: Duplicate Sentence Detection (Binary Classification: equivalent vs. not equivalent)
- **Original Source**: Microsoft Research / GLUE Benchmark
- **Paper**: Dolan & Brockett (2005), *"Automatically Constructing a Corpus of Sentential Paraphrases"*, IWP 2005.
- **Direct Distribution URL**: `https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt` and `test.txt`
- **Local Location**: `data/raw/mrpc/` (Train: 4,076 samples, Test: 1,725 samples)
- **Preprocessed Schema**: `[sentence1, sentence2] -> {0: not equivalent, 1: equivalent}`

### 2. JFLEG (JHU FLuency-Extended GUG)
- **Task**: Grammar Correction (Text Generation)
- **Authors**: Napoles, Sakaguchi, & Tetreault (2017)
- **Paper**: *"JFLEG: A Fluency Corpus and Benchmark for Grammatical Error Correction"*, EACL 2017.
- **Direct Distribution URL**: `https://raw.githubusercontent.com/keisks/jfleg/master/` (`dev/` and `test/`)
- **Local Location**: `data/raw/jfleg/` (Dev: 754 samples, Test: 747 samples; each with 4 human correction references `ref0` through `ref3`)
- **Preprocessed Schema**: `sentence -> list of 4 gold corrections`

### 3. HSOL (Hate Speech and Offensive Language)
- **Task**: Hate Content Detection (Binary Classification: harmful vs. clean/neither)
- **Authors**: Davidson, Warmsley, Macy, & Weber (2017)
- **Paper**: *"Automated Hate Speech Detection and the Problem of Offensive Language"*, ICWSM 2017.
- **Direct Distribution URL**: `https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv`
- **Local Location**: `data/raw/hsol/` (24,783 annotated tweets)
- **Preprocessed Schema**: `tweet -> {0: neither/clean, 1: hateful or offensive}`

### 4. RTE (Recognizing Textual Entailment)
- **Task**: Natural Language Inference (Binary Classification: entailment vs. not entailment)
- **Source**: PASCAL RTE Challenges / GLUE Benchmark (Wang et al., ICLR 2019)
- **Direct Distribution URL**: `https://dl.fbaipublicfiles.com/glue/data/RTE.zip`
- **Local Location**: `data/raw/rte/` (Train: 2,490 samples, Dev: 277 samples)
- **Preprocessed Schema**: `[sentence1, sentence2] -> {0: entailment, 1: not entailment}`

### 5. SST-2 (Stanford Sentiment Treebank v2)
- **Task**: Sentiment Analysis (Binary Classification: negative vs. positive)
- **Authors**: Socher et al. (2013), *"Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank"*, EMNLP 2013.
- **Direct Distribution URL**: `https://dl.fbaipublicfiles.com/glue/data/SST-2.zip`
- **Local Location**: `data/raw/sst2/` (Train: 67,349 samples, Dev: 872 samples, Test: 1,821 samples)
- **Preprocessed Schema**: `sentence -> {0: negative, 1: positive}`

### 6. SMS Spam Collection
- **Task**: Spam Detection (Binary Classification: ham vs. spam)
- **Authors**: Almeida, Hidalgo, & Yamakami (2011), *"Contributions to the Study of SMS Spam Filtering"*, ACM DOCENG 2011.
- **Direct Distribution URL**: `https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv` (UCI ML Repository mirror)
- **Local Location**: `data/raw/sms_spam/` (5,574 labeled messages)
- **Preprocessed Schema**: `sms -> {0: ham, 1: spam}`

### 7. Annotated Gigaword
- **Task**: Text Summarization (Text Generation)
- **Authors**: Rush, Chopra, & Weston (2015), *"A Neural Attention Model for Abstractive Sentence Summarization"*, EMNLP 2015.
- **Official Open-Prompt-Injection Archive**: `https://drive.google.com/uc?export=download&id=1USoQ8lJgN8kAWnUnRrupMGrPMLlDVqlV` (HarvardNLP sent-summary format)
- **Local Location**: `data/raw/gigaword/` (Dev: 189,651 sentence-summary pairs, Test: 1,951 pairs)
- **Preprocessed Schema**: `document -> summary`

---

## 3. Pretrained Models & Semantic Embedding Stack

1. **Lightweight Detection Transformer Backbone**:
   - `distilbert-base-uncased` (Sanh et al., 2019, NeurIPS EMC2 Workshop)
   - Chosen for CPU-resilient inference latency and low memory footprint (~66M parameters).
2. **Context-Aware Semantic Encoder**:
   - `sentence-transformers/all-MiniLM-L6-v2` (Reimers & Gurevych, EMNLP 2019)
   - Dimensions: 384, chosen for dense cosine-similarity comparison between target task instruction and suspicious input spans.

---

## 4. Multimodal & OCR Engine Stack

1. **PyPDF** (`pypdf-6.19.0`): Clean digital PDF text extraction.
2. **PyTesseract / Tesseract OCR engine**: Optical character recognition for scanned PDFs, PNGs, and JPEG documents.
