# 🔬 Auto-Severity Classification — Text Preprocessing Pipeline

Pipeline pemrosesan teks otomatis untuk mengklasifikasikan tingkat keparahan temuan/bug pada sistem **Web Simata**.

## 📋 Deskripsi

Modul ini merupakan **Tahap 1** dari sistem Auto-Severity Classification yang memproses teks mentah "Log Temuan" dari tester menjadi token bersih yang siap digunakan untuk modelling (Tahap 2).

## 🏗️ Arsitektur Pipeline

```
Input Text (raw "Log Temuan")
    │
    ├─ Tahap 1: Word Segmentation & Boundary Splitting  (symspellpy + regex)
    ├─ Tahap 2: Case Folding / Lowercasing               (str.lower())
    ├─ Tahap 3: Regex Cleaning                            (re — NFA/DFA)
    ├─ Tahap 4: Dynamic Slang Normalization               (HashMap Lookup)
    └─ Tahap 5: Filtering & Custom Stopwords              (PySastrawi + Whitelist)
    │
Output Text (clean tokens)
```

## 📂 Struktur Proyek

```
implement-automation/
├── data/
│   ├── raw/                          # Dataset CSV mentah
│   ├── processed/                    # Output hasil preprocessing
│   └── dictionaries/                 # Kamus slang & frequency dict
├── preprocessing/                    # Modul preprocessing (5 tahap)
│   ├── __init__.py
│   ├── pipeline.py                   # Orchestrator
│   ├── word_segmenter.py             # Tahap 1
│   ├── case_folder.py                # Tahap 2
│   ├── regex_cleaner.py              # Tahap 3
│   ├── slang_normalizer.py           # Tahap 4
│   └── stopword_filter.py           # Tahap 5
├── main.py                           # CLI Entry Point
├── config.py                         # Konfigurasi global
├── requirements.txt                  # Dependencies
└── README.md                         # Dokumentasi (file ini)
```

## 🚀 Instalasi & Penggunaan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Mode Batch (Proses CSV)

```bash
python main.py --input data/raw/dataset_temuan_sample.csv --output data/processed/dataset_cleaned.csv
```

Dengan output verbose per tahap:

```bash
python main.py --input data/raw/dataset_temuan_sample.csv --verbose
```

### 3. Mode Interaktif (Testing Manual)

```bash
python main.py --interactive
```

### 4. Penggunaan sebagai Library

```python
from preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline()

# Proses satu teks
result = pipeline.preprocess_text("1)tidak bisa login, error500 muncul")
print(result)

# Proses DataFrame
import pandas as pd
df = pd.read_csv("data/raw/dataset.csv")
df_clean = pipeline.preprocess_dataframe(df)
```

## ⚙️ Konfigurasi

Edit `config.py` untuk mengatur:

- **Path** file data & dictionary
- **Whitelist** kata negasi & severity indicators
- **Toggle** on/off setiap tahap pipeline
- **SymSpell** parameters

## 📚 Teknologi

| Tahap | Teknologi | Algoritma |
|-------|-----------|-----------|
| Word Segmentation | `symspellpy`, `re` | Viterbi Algorithm, Regex Boundary Matching |
| Case Folding | `str.lower()` | Unicode/ASCII Case Mapping |
| Regex Cleaning | `re` | NFA/DFA Pattern Matching Engine |
| Slang Normalization | `pandas`, `dict` | Hash Map Lookup O(1) |
| Stopword Filtering | `PySastrawi` | Set Membership Filtering (Hash Set) |
