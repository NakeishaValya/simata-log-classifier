"""
config.py — Konfigurasi Global untuk Pipeline Preprocessing
============================================================
Menyimpan semua konstanta path dan parameter konfigurasi
yang digunakan oleh seluruh modul preprocessing.

Pipeline 2 Tahap:
  1. Text Cleaning & HTML Preservation (Regex Engine)
  2. Feature Extraction & Embedding (IndoBERT Encoder)
"""

import os

# ======================================================================
# PATH CONFIGURATION
# ======================================================================
# Base directory proyek (root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Direktori data
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# ======================================================================
# COLUMN CONFIGURATION
# ======================================================================
# Nama kolom pada dataset CSV
INPUT_COLUMN = "Log Temuan"           # Kolom input teks mentah
LABEL_COLUMN = "Kategori"            # Kolom label severity
OUTPUT_COLUMN = "cleaned_text"       # Kolom output hasil preprocessing
EMBEDDING_COLUMN = "embedding"       # Kolom referensi embedding (file .npy)

# ======================================================================
# PIPELINE CONFIGURATION
# ======================================================================
# Toggle untuk mengaktifkan/menonaktifkan tahap preprocessing
PIPELINE_STAGES = {
    "text_cleaning": True,         # Tahap 1: Text Cleaning & HTML Preservation
    "feature_extraction": True,    # Tahap 2: IndoBERT Feature Extraction
}

# ======================================================================
# INDOBERT CONFIGURATION
# ======================================================================
# Model IndoBERT dari HuggingFace
INDOBERT_MODEL_NAME = "indobenchmark/indobert-base-p1"
INDOBERT_MAX_LENGTH = 128   # Panjang maksimum token (128 cukup untuk bug reports)

# ======================================================================
# LOGGING CONFIGURATION
# ======================================================================
LOG_LEVEL = "INFO"                # Level logging: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
