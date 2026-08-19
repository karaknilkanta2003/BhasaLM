# Monolingual Transformer Language Models in Two Indian Languages (LMA Assignment)

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

This repository contains the complete implementation, pipelines, and evaluation for **Phase 1 (Data Collection, Preprocessing & Tokenization)** of two standalone ~25M parameter monolingual Transformer Language Models:
- **Model H (High-Resource):** **Hindi (`hi`)** — **504M Subword Tokens**
- **Model L (Low-Resource):** **Assamese (`as`)** — **501M Subword Tokens**

---

DataSet-Link -- https://drive.google.com/drive/folders/1xSOCnXsHGA3vjKqUEfuPcypovd9kG1P1?usp=sharing

## 1. Data Sources & Composition

All training corpora consist of 100% human-authored text (zero synthetic data). Non-language English/Latin words (`[a-zA-Z]+`) are stripped while all numerals (`0-9`, `०-९`, `০-৯`) and punctuation/Dandas (`।`) are strictly preserved.

### A. Hindi (Model H — 504,000,634 Tokens Total)
- **Manual Stream (20.24% = 102,000,531 Tokens):**
  - **Source:** Official Wikimedia Hindi XML Database Dump (`wikimedia/wikipedia` snapshot `20231101.hi`) + Sangraha Verified Scanned Docs.
  - **Custom WikiExtractor:** Strips infoboxes, wiki templates, internal links, and category metadata.
  - **Priority Cultural Themes:** *Mahabharata*, *Ramayana*, *Puranas*, *Vedas*, Indian history, philosophy, and classical Hindi literature (Munshi Premchand, Tulsidas, Kalidas).
  - **Metadata Tag:** `{"source_type": "manual", "source_name": "custom_hindi_wikidump_extractor"}`
- **Downloaded Stream (79.76% = 402,000,103 Tokens):**
  - **Source:** **AI4Bharat IndicCorp v2** (`ai4bharat/IndicCorpV2`, split `hin_Deva`) + Sangraha Verified Hindi.
  - **Content:** Cleaned web crawls, public news archives, encyclopedias, and educational articles.
  - **Metadata Tag:** `{"source_type": "downloaded", "source_name": "ai4bharat_indiccorp_v2_hin"}`

---

### B. Assamese (Model L — 501,000,349 Tokens Total)
- **Manual Stream (20.16% = 101,000,294 Tokens):**
  - **Source:** Web-scraped **Assamese Wikipedia** (`wikimedia/wikipedia` `20231101.as`) + **AI4Bharat Sangraha Verified Scanned Books/OCR docs** (`ai4bharat/sangraha`, `verified`, `asm`) + Regional literature portals.
  - **Content:** Scanned books OCR, folklore (*Burhi Aair Sadhu*), regional literature (Lakshminath Bezbaroa, Srimanta Sankardev), and encyclopedia articles.
  - **Metadata Tag:** `{"source_type": "manual", "source_name": "assamese_web_scraped_wikipedia"}`
- **Downloaded Stream (79.84% = 400,000,055 Tokens):**
  - **Source:** **AI4Bharat IndicCorp v2** (`asm_Beng`) + **Sangraha Web Crawl** (`unverified`, `asm`) + **Deepjyoti120 Regional News Articles**.
  - **Content:** Large-scale web text, regional news portals (*Asamiya Pratidin*, *Amar Asom*, *Dainik Janambhumi*).
  - **Metadata Tag:** `{"source_type": "downloaded"}`

---

## 2. Repository File Structure

```
.
├── common/                               # Shared utilities and verification tools
│   ├── __init__.py
│   ├── utils.py                          # Logging and seed utilities
│   └── verify_tokens.py                  # Programmatic 500M token verification script
├── hindi/                                # Hindi (Model H) pipeline modules
│   ├── data/
│   │   ├── collect.py                    # Downloaded IndicCorp v2 stream collector
│   │   ├── collect_manual_wiki.py        # 102M Manual WikiDump & Epics collector
│   │   ├── clean.py                      # Devanagari cleaner + Latin word stripper
│   │   └── split.py                      # 90/5/5 document-level splitter (0.0% leakage)
│   ├── tokenizer/
│   │   ├── train.py                      # Standalone SentencePiece BPE trainer (16k vocab)
│   │   ├── evaluate.py                   # Token frequency, fertility & UNK evaluator
│   │   ├── hindi_sp.model                # Trained binary tokenizer model
│   │   └── hindi_sp.vocab                # Human-readable 16,000 vocab file
│   └── run_phase1.py                     # Local single-entry point runner for Hindi
├── assamese/                             # Assamese (Model L) pipeline modules
│   ├── data/
│   │   ├── collect.py                    # Downloaded IndicCorp & Sangraha collector
│   │   ├── collect_manual.py             # 101M Manual Scraped & Books OCR collector
│   │   ├── clean.py                      # Assamese cleaner (ৰ, ৱ) + Latin word stripper
│   │   └── split.py                      # 90/5/5 document-level splitter (0.0% leakage)
│   ├── tokenizer/
│   │   ├── train.py                      # Standalone SentencePiece BPE trainer (16k vocab)
│   │   ├── evaluate.py                   # Token frequency, fertility & UNK evaluator
│   │   ├── assamese_sp.model             # Trained binary tokenizer model
│   │   └── assamese_sp.vocab             # Human-readable 16,000 vocab file
│   └── run_phase1.py                     # Local single-entry point runner for Assamese
├── kaggle_runners/                       # Self-contained Kaggle scripts (pushed via Kaggle CLI)
│   ├── hindi/
│   │   ├── kernel-metadata.json          # Kaggle metadata (nilkantakarak/lma-hindi-phase-1-data-and-tokenizer)
│   │   └── main.py                       # Self-contained 504M Hindi pipeline
│   └── assamese/
│       ├── kernel-metadata.json          # Kaggle metadata (nilkantakarak/lma-assamese-phase-1-data-and-tokenizer)
│       └── main.py                       # Self-contained 501M Assamese pipeline
└── report/
    └── phase1/                           # Phase 1 Deliverables
        ├── report.md                     # Comprehensive Phase 1 Report
        └── AI_USAGE.md                   # AI Usage Disclosure
```

---

## 3. How to Run & Reproduce

### 3.1 Environment Setup
```bash
pip install sentencepiece datasets requests tqdm beautifulsoup4
```

### 3.2 Running on Kaggle (Recommended via Kaggle CLI)
Authenticate and push the self-contained kernels directly to Kaggle GPU/CPU servers:

```powershell
# 1. Set Kaggle token
$env:KAGGLE_API_TOKEN = (kaggle auth print-access-token)

# 2. Push and execute Hindi 504M Pipeline on Kaggle
kaggle kernels push -p kaggle_runners/hindi

# 3. Push and execute Assamese 501M Pipeline on Kaggle
kaggle kernels push -p kaggle_runners/assamese

# 4. Check live status
kaggle kernels status nilkantakarak/lma-hindi-phase-1-data-and-tokenizer
kaggle kernels status nilkantakarak/lma-assamese-phase-1-data-and-tokenizer
```

### 3.3 Running Locally
```bash
# Run Hindi Pipeline (Data collection, cleaning, splitting, tokenizer training)
python hindi/run_phase1.py

# Run Assamese Pipeline
python assamese/run_phase1.py
```

### 3.4 Programmatic Token Verification
Run the verification script to inspect dataset token counts, manifests, and test UNK rates:

```bash
# Verify Assamese Dataset & Tokenizer
python common/verify_tokens.py \
  --model assamese_501m_output/tokenizer/assamese_sp.model \
  --data_dir assamese_501m_output/data

# Verify Hindi Dataset & Tokenizer
python common/verify_tokens.py \
  --model hindi_504m_output/tokenizer/hindi_sp.model \
  --data_dir hindi_504m_output/data
```

---

## 4. Main Results & Verified Metrics

| Metric | Hindi (Model H — High-Resource) | Assamese (Model L — Low-Resource) | Compliance Status |
|---|---|---|---|
| **Manual Component ($\ge 20\%$ Quota)** | **102,000,531 Tokens (20.24%)** | **101,000,294 Tokens (20.16%)** | **$\ge 20\%$ Quota Strictly Met** |
| **Downloaded Component ($\approx 80\%$)** | **402,000,103 Tokens (79.76%)** | **400,000,055 Tokens (79.84%)** | **Compliant** |
| **Total Resultant Corpus** | **504,000,634 Tokens** | **501,000,349 Tokens** | **$\ge 500\text{M}$ Tokens Target Met** |
| **Total Words** | **377,712,325 Words** | **310,848,769 Words** | **Cleaned & Processed** |
| **Total Unique Documents** | **4,385,085 Documents** | **5,145,096 Documents** | **100% Unique (0 MD5 Duplicates)** |
| **Non-Language Words (English)** | **Stripped** (`re.sub(r"[a-zA-Z]+", " ", text)`) | **Stripped** (`re.sub(r"[a-zA-Z]+", " ", text)`) | **Compliant with FAQ rule** |
| **Digits (0-9, Indic Digits)** | **Preserved** | **Preserved** | **Compliant with FAQ rule** |
| **Train Split (90%)** | 3,946,602 docs (~453.6M tok) | 4,630,484 docs (~450.9M tok) | **0.0% Data Leakage** |
| **Val Split (5%)** | 219,030 docs (~25.2M tok) | 257,486 docs (~25.0M tok) | **0.0% Data Leakage** |
| **Test Split (5%)** | 219,453 docs (~25.2M tok) | 257,126 docs (~25.1M tok) | **0.0% Data Leakage** |
| **Tokenizer Architecture** | SentencePiece BPE (16,000 Vocab) | SentencePiece BPE (16,000 Vocab) | **Trained from scratch** |
| **Fertility (Subwords / Word)** | **1.334** | **1.612** | **Optimal subword compression** |
| **Test UNK Rate** | **0.0%** (via byte-fallback) | **0.0%** (via byte-fallback) | **Zero out-of-vocabulary crashes** |
