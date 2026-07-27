"""
config.py — Konfigurasi Global untuk Pipeline Preprocessing
============================================================
Menyimpan semua konstanta path, whitelist kata, dan parameter
konfigurasi yang digunakan oleh seluruh modul preprocessing.
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
DICTIONARIES_DIR = os.path.join(DATA_DIR, "dictionaries")

# Path file spesifik
SLANG_DICT_PATH = os.path.join(DICTIONARIES_DIR, "kamus_slang.csv")
FREQ_DICT_PATH = os.path.join(DICTIONARIES_DIR, "id_freq_dict.txt")

# ======================================================================
# COLUMN CONFIGURATION
# ======================================================================
# Nama kolom pada dataset CSV
INPUT_COLUMN = "Log Temuan"           # Kolom input teks mentah
LABEL_COLUMN = "Kategori"            # Kolom label severity
OUTPUT_COLUMN = "cleaned_text"       # Kolom output hasil preprocessing

# ======================================================================
# SEVERITY WHITELIST — Kata-kata yang TIDAK BOLEH dihapus oleh stopword filter
# ======================================================================
# Kata negasi: membalikkan makna kalimat, krusial untuk severity classification
NEGATION_WORDS = {
    "tidak", "belum", "gagal", "bukan", "jangan", "tanpa",
    "tak", "tiada", "enggan", "mustahil", "batal",
}

# Kata indikator severity: menunjukkan tingkat keparahan bug
SEVERITY_INDICATORS = {
    # Fatal / Critical indicators
    "error", "crash", "hang", "freeze", "fatal", "critical",
    "down", "mati", "putus", "corrupt", "rusak", "hilang",
    "timeout", "exception", "overflow", "null", "undefined",
    "denied", "forbidden", "unauthorized", "stuck", "broken",

    # Mayor indicators
    "gagal", "salah", "fail", "failed", "failure", "invalid",
    "stop", "required", "bug", "warning", "blank", "kosong",
    "missing", "duplikat", "duplicate",

    # Minor indicators
    "lambat", "lama", "slow", "delay", "inkonsisten",
    "inconsistent",

    # Kosmetik indicators
    "blur", "pecah", "kecil", "besar", "rapat", "sempit",
    "flat", "jelek",
}

# Gabungan whitelist: kata-kata yang HARUS dipertahankan
WHITELIST_WORDS = NEGATION_WORDS | SEVERITY_INDICATORS

# ======================================================================
# PIPELINE CONFIGURATION
# ======================================================================
# Toggle untuk mengaktifkan/menonaktifkan tahap preprocessing
PIPELINE_STAGES = {
    "word_segmentation": True,    # Tahap 1
    "case_folding": True,         # Tahap 2
    "regex_cleaning": True,       # Tahap 3
    "slang_normalization": True,  # Tahap 4
    "stopword_filtering": True,   # Tahap 5
}

# ======================================================================
# SYMSPELLPY CONFIGURATION
# ======================================================================
# Konfigurasi untuk word segmentation via symspellpy
SYMSPELL_MAX_EDIT_DISTANCE = 0    # 0 = pure segmentation tanpa koreksi ejaan
SYMSPELL_PREFIX_LENGTH = 7        # Panjang prefix untuk indexing

# ======================================================================
# LOGGING CONFIGURATION
# ======================================================================
LOG_LEVEL = "INFO"                # Level logging: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
