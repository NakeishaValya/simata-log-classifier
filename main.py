"""
main.py — Entry Point CLI untuk Pipeline Preprocessing
=======================================================
Script utama untuk menjalankan pipeline preprocessing 2 tahap.

Cara pakai:
  $ python main.py

Flow:
  1. Jalankan script
  2. Pilih file CSV melalui dialog browser
  3. File CSV dicopy ke folder data/raw/
  4. Tahap 1: Text Cleaning & HTML Preservation
  5. Tahap 2: Feature Extraction via IndoBERT
  6. Hasil disimpan ke folder data/processed/
     - *_cleaned.csv  : teks bersih
     - *_embeddings.npy: vektor embedding 768-dim
"""

import logging
import os
import shutil
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd

import config
from preprocessing import PreprocessingPipeline


def setup_logging():
    """
    Konfigurasi logging untuk seluruh aplikasi.
    """
    log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )


def print_header():
    """Tampilkan header aplikasi."""
    print(f"\n{'='*60}")
    print(f"  PREPROCESSING PIPELINE (2 Tahap)")
    print(f"  Auto-Severity Classification (Web Simata)")
    print(f"{'='*60}")
    print(f"  Tahap 1: Text Cleaning & HTML Preservation")
    print(f"  Tahap 2: Feature Extraction (IndoBERT)")
    print(f"{'='*60}")
    print(f"  Pilih file CSV melalui dialog yang muncul.")
    print(f"  Hasil disimpan ke data/processed/")
    print(f"{'='*60}\n")


def get_csv_path():
    """
    Buka file browser dialog untuk memilih file CSV.

    Returns:
        str: Path absolut ke file CSV yang valid.
    """
    # Inisialisasi tkinter root (hidden)
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    while True:
        print("  [i] Membuka file browser... Pilih file CSV.")

        csv_path = filedialog.askopenfilename(
            title="Pilih File CSV Data Temuan",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialdir=os.path.expanduser("~"),
        )

        # User menekan Cancel
        if not csv_path:
            print("  [!] Tidak ada file yang dipilih.")
            retry = messagebox.askyesno(
                "Tidak Ada File",
                "Tidak ada file yang dipilih.\nApakah ingin memilih file lagi?",
            )
            if not retry:
                root.destroy()
                print("  [!] Dibatalkan oleh user.")
                sys.exit(0)
            continue

        csv_path = os.path.abspath(csv_path)

        # Validasi bisa dibaca sebagai CSV
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", nrows=0)
        except Exception as e:
            messagebox.showerror(
                "Gagal Membaca CSV",
                f"File tidak bisa dibaca sebagai CSV:\n{e}",
            )
            continue

        # Validasi kolom yang dibutuhkan
        if config.INPUT_COLUMN not in df.columns:
            messagebox.showerror(
                "Kolom Tidak Ditemukan",
                f"Kolom '{config.INPUT_COLUMN}' tidak ditemukan.\n\n"
                f"Kolom yang tersedia:\n{list(df.columns)}\n\n"
                f"Pastikan CSV memiliki kolom '{config.INPUT_COLUMN}'.",
            )
            continue

        root.destroy()
        return csv_path


def copy_to_raw(source_path):
    """
    Copy file CSV ke folder data/raw/.

    Args:
        source_path: Path file CSV sumber.

    Returns:
        str: Path file CSV di folder data/raw/.
    """
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

    filename = os.path.basename(source_path)
    raw_path = os.path.join(config.RAW_DATA_DIR, filename)

    # Jika file sudah ada di raw, tambahkan suffix
    if os.path.exists(raw_path) and os.path.abspath(source_path) != os.path.abspath(raw_path):
        name, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(raw_path):
            raw_path = os.path.join(config.RAW_DATA_DIR, f"{name}_{counter}{ext}")
            counter += 1

    # Copy file (skip jika source sudah di raw)
    if os.path.abspath(source_path) != os.path.abspath(raw_path):
        shutil.copy2(source_path, raw_path)

    return raw_path


def run_preprocessing(raw_path):
    """
    Jalankan pipeline preprocessing pada file CSV.

    Args:
        raw_path: Path file CSV di folder data/raw/.

    Returns:
        tuple: (result_df, output_path)
    """
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    # Tentukan nama output berdasarkan nama file input
    input_name = os.path.splitext(os.path.basename(raw_path))[0]
    output_filename = f"{input_name}_cleaned.csv"
    output_path = os.path.join(config.PROCESSED_DATA_DIR, output_filename)

    print(f"\n{'='*60}")
    print(f"  MEMPROSES DATA")
    print(f"{'='*60}")
    print(f"  Input  : {raw_path}")
    print(f"  Output : {output_path}")
    print(f"{'='*60}\n")

    # Inisialisasi dan jalankan pipeline
    pipeline = PreprocessingPipeline()
    result_df = pipeline.preprocess_csv(
        input_path=raw_path,
        output_path=output_path,
    )

    return result_df, output_path


def print_summary(result_df, raw_path, output_path):
    """
    Tampilkan ringkasan hasil preprocessing.
    """
    print(f"\n{'='*60}")
    print(f"  RINGKASAN HASIL")
    print(f"{'='*60}")
    print(f"  Total baris diproses   : {len(result_df)}")
    print(f"  Kolom output           : '{config.OUTPUT_COLUMN}'")
    print(f"  File raw (input)       : {raw_path}")
    print(f"  File processed (output): {output_path}")

    # Info embedding
    npy_path = output_path.replace(".csv", "_embeddings.npy")
    if os.path.exists(npy_path):
        import numpy as np
        emb = np.load(npy_path)
        print(f"  Embedding file         : {npy_path}")
        print(f"  Embedding shape        : {emb.shape}")
    else:
        print(f"  Embedding              : Tidak tersedia")

    print(f"{'='*60}")

    # Tampilkan sample hasil (5 baris pertama)
    print(f"\n  Sample Hasil (5 baris pertama):")
    print(f"  {'-'*56}")
    for idx, row in result_df.head(5).iterrows():
        raw = str(row.get(config.INPUT_COLUMN, ""))[:50]
        clean = str(row.get(config.OUTPUT_COLUMN, ""))[:50]
        label = str(row.get(config.LABEL_COLUMN, "N/A"))
        print(f"  [{label:^8}]")
        print(f"    RAW   : {raw}...")
        print(f"    CLEAN : {clean}...")
        print(f"  {'-'*56}")

    print(f"\n  [OK] Preprocessing selesai!\n")


def main():
    """
    Entry point utama.
    Flow: Input CSV → Copy ke raw → Preprocessing (2 tahap) → Simpan ke processed.
    """
    # Setup logging
    setup_logging()

    # Tampilkan header
    print_header()

    # Step 1: Minta path file CSV dari user
    csv_path = get_csv_path()
    print(f"\n  [OK] File ditemukan: {csv_path}")

    # Step 2: Copy ke folder data/raw/
    raw_path = copy_to_raw(csv_path)
    print(f"  [OK] File dicopy ke : {raw_path}")

    # Step 3: Jalankan preprocessing (2 tahap)
    result_df, output_path = run_preprocessing(raw_path)

    # Step 4: Tampilkan ringkasan
    print_summary(result_df, raw_path, output_path)


if __name__ == "__main__":
    main()
