"""
main.py — Entry Point CLI (Full Pipeline)
==========================================
Script utama untuk menjalankan full pipeline:
  Preprocessing → Training → Testing

Cara pakai:
  $ python main.py

Flow:
  1. Menu: Upload file baru ATAU gunakan data yang sudah ada
  2. Jika upload:
     a. Pilih CSV TRAIN via dialog browser
     b. Pilih CSV TEST via dialog browser
     c. File lama di raw/ OTOMATIS dihapus, diganti file baru
     d. Preprocessing kedua file → simpan ke processed/
     e. Konfirmasi: lanjut training?
  3. Jika gunakan data lama:
     a. Langsung training dari data processed/ yang sudah ada
  4. Training: IndoBERT + LinearSVC + XGBoost + Ensemble
  5. Testing: Evaluasi pada test set, tampilkan perbandingan
  6. Save models ke models/
"""

import glob
import logging
import os
import shutil
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd

import config
from preprocessing import PreprocessingPipeline


# ======================================================================
# CONSTANTS — Nama standar file train/test
# ======================================================================
RAW_TRAIN_NAME = "train_data.csv"
RAW_TEST_NAME = "test_data.csv"
PROCESSED_TRAIN_NAME = "train_cleaned.csv"
PROCESSED_TEST_NAME = "test_cleaned.csv"


# ======================================================================
# Helpers
# ======================================================================
def setup_logging():
    """Konfigurasi logging."""
    log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )


def print_header():
    """Tampilkan header aplikasi."""
    print(f"\n{'='*60}")
    print(f"  AUTO-SEVERITY CLASSIFICATION PIPELINE")
    print(f"  Preprocessing + Training + Testing")
    print(f"{'='*60}")
    print(f"  Tahap 1: Text Cleaning & HTML Preservation")
    print(f"  Tahap 2: Feature Extraction (IndoBERT)")
    print(f"  Tahap 3: Model Training & Ensemble")
    print(f"{'='*60}\n")


def print_menu():
    """Tampilkan menu utama dan ambil pilihan user."""
    print(f"  Pilih opsi:")
    print(f"  [1] Upload file CSV baru (Train + Test)")
    print(f"  [2] Gunakan data yang sudah ada (langsung training)")
    print(f"  [0] Keluar")
    print()

    while True:
        choice = input("  Masukkan pilihan (0/1/2): ").strip()
        if choice in ("0", "1", "2"):
            return choice
        print("  [!] Pilihan tidak valid. Masukkan 0, 1, atau 2.")


# ======================================================================
# File Dialog
# ======================================================================
def pick_csv_file(title: str, root: tk.Tk) -> str:
    """
    Buka file browser dialog untuk memilih satu file CSV.

    Args:
        title: Judul dialog.
        root:  tkinter root (hidden).

    Returns:
        Path absolut ke file CSV, atau None jika dibatalkan.
    """
    while True:
        print(f"  [i] Membuka file browser... {title}")

        csv_path = filedialog.askopenfilename(
            title=title,
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialdir=os.path.expanduser("~"),
        )

        if not csv_path:
            retry = messagebox.askyesno(
                "Tidak Ada File",
                f"Tidak ada file yang dipilih untuk '{title}'.\n"
                "Apakah ingin memilih file lagi?",
            )
            if not retry:
                return None
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

        # Validasi kolom
        if config.INPUT_COLUMN not in df.columns:
            messagebox.showerror(
                "Kolom Tidak Ditemukan",
                f"Kolom '{config.INPUT_COLUMN}' tidak ditemukan.\n\n"
                f"Kolom yang tersedia:\n{list(df.columns)}\n\n"
                f"Pastikan CSV memiliki kolom '{config.INPUT_COLUMN}'.",
            )
            continue

        return csv_path


# ======================================================================
# Upload & Preprocessing
# ======================================================================
def clear_raw_folder():
    """Hapus semua file di data/raw/ (file lama diganti yang baru)."""
    if os.path.exists(config.RAW_DATA_DIR):
        for f in glob.glob(os.path.join(config.RAW_DATA_DIR, "*")):
            os.remove(f)
        print("  [OK] File lama di data/raw/ dihapus.")


def clear_processed_folder():
    """Hapus semua file di data/processed/ (hasil lama diganti)."""
    if os.path.exists(config.PROCESSED_DATA_DIR):
        for f in glob.glob(os.path.join(config.PROCESSED_DATA_DIR, "*")):
            os.remove(f)
        print("  [OK] File lama di data/processed/ dihapus.")


def copy_to_raw(source_path: str, target_name: str) -> str:
    """
    Copy file CSV ke data/raw/ dengan nama standar.

    Args:
        source_path: Path file CSV sumber.
        target_name: Nama file target (train_data.csv / test_data.csv).

    Returns:
        Path file di data/raw/.
    """
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    raw_path = os.path.join(config.RAW_DATA_DIR, target_name)
    shutil.copy2(source_path, raw_path)
    return raw_path


def run_preprocessing(raw_path: str, output_name: str):
    """
    Jalankan pipeline preprocessing pada satu file CSV.

    Args:
        raw_path:    Path file CSV di data/raw/.
        output_name: Nama file output (train_cleaned.csv / test_cleaned.csv).

    Returns:
        tuple: (result_df, output_path)
    """
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(config.PROCESSED_DATA_DIR, output_name)

    pipeline = PreprocessingPipeline()
    result_df = pipeline.preprocess_csv(
        input_path=raw_path, output_path=output_path,
    )
    return result_df, output_path


def upload_and_preprocess():
    """
    Flow upload file baru:
    1. Pilih CSV TRAIN
    2. Pilih CSV TEST
    3. Hapus file lama di raw/ dan processed/
    4. Copy file baru ke raw/
    5. Preprocess keduanya → simpan ke processed/

    Returns:
        tuple: (train_processed_path, test_processed_path) atau None jika dibatalkan.
    """
    # Buka dialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print(f"\n{'='*60}")
    print(f"  UPLOAD FILE CSV BARU")
    print(f"{'='*60}\n")

    # Pilih file TRAIN
    train_source = pick_csv_file("Pilih CSV DATA TRAIN", root)
    if not train_source:
        root.destroy()
        print("  [!] Upload dibatalkan.")
        return None
    print(f"  [OK] File TRAIN: {train_source}")

    # Pilih file TEST
    test_source = pick_csv_file("Pilih CSV DATA TEST", root)
    if not test_source:
        root.destroy()
        print("  [!] Upload dibatalkan.")
        return None
    print(f"  [OK] File TEST : {test_source}")

    root.destroy()

    # Hapus file lama
    clear_raw_folder()
    clear_processed_folder()

    # Copy ke raw/
    raw_train = copy_to_raw(train_source, RAW_TRAIN_NAME)
    raw_test = copy_to_raw(test_source, RAW_TEST_NAME)
    print(f"  [OK] TRAIN dicopy ke: {raw_train}")
    print(f"  [OK] TEST  dicopy ke: {raw_test}")

    # Preprocessing TRAIN
    print(f"\n{'='*60}")
    print(f"  PREPROCESSING DATA TRAIN")
    print(f"{'='*60}")
    df_train, train_out = run_preprocessing(raw_train, PROCESSED_TRAIN_NAME)
    print(f"  [OK] Train processed: {train_out} ({len(df_train)} baris)")

    # Preprocessing TEST
    print(f"\n{'='*60}")
    print(f"  PREPROCESSING DATA TEST")
    print(f"{'='*60}")
    df_test, test_out = run_preprocessing(raw_test, PROCESSED_TEST_NAME)
    print(f"  [OK] Test processed : {test_out} ({len(df_test)} baris)")

    # Ringkasan
    print(f"\n{'='*60}")
    print(f"  PREPROCESSING BERHASIL!")
    print(f"{'='*60}")
    print(f"  Train: {len(df_train)} baris -> {train_out}")
    print(f"  Test : {len(df_test)} baris -> {test_out}")
    print(f"{'='*60}\n")

    return train_out, test_out


# ======================================================================
# Check existing data
# ======================================================================
def check_existing_data():
    """
    Cek apakah data processed untuk train dan test sudah ada.

    Returns:
        tuple: (train_path, test_path) jika ada, None jika tidak.
    """
    train_path = os.path.join(config.PROCESSED_DATA_DIR, PROCESSED_TRAIN_NAME)
    test_path = os.path.join(config.PROCESSED_DATA_DIR, PROCESSED_TEST_NAME)

    if os.path.exists(train_path) and os.path.exists(test_path):
        df_train = pd.read_csv(train_path, encoding="utf-8", nrows=0)
        df_test = pd.read_csv(test_path, encoding="utf-8", nrows=0)

        # Validasi kolom
        if config.OUTPUT_COLUMN in df_train.columns and config.OUTPUT_COLUMN in df_test.columns:
            return train_path, test_path

    return None


def confirm_training() -> bool:
    """Tanya user apakah ingin melanjutkan ke training."""
    while True:
        answer = input("  Lanjutkan ke training? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  [!] Masukkan 'y' atau 'n'.")


# ======================================================================
# Training
# ======================================================================
def run_training(train_csv: str, test_csv: str):
    """
    Jalankan training menggunakan data yang sudah dipreprocess.

    Args:
        train_csv: Path ke CSV train yang sudah dipreprocess.
        test_csv:  Path ke CSV test yang sudah dipreprocess.
    """
    from trainer import ModelTrainer

    print(f"\n{'='*60}")
    print(f"  MODEL TRAINING & TESTING")
    print(f"{'='*60}")
    print(f"  Train data: {train_csv}")
    print(f"  Test data : {test_csv}")
    print(f"{'='*60}")

    trainer = ModelTrainer(train_csv=train_csv, test_csv=test_csv)
    trainer.run()


# ======================================================================
# Main
# ======================================================================
def main():
    """
    Entry point utama.

    Menu:
      [1] Upload CSV baru (train + test) → preprocess → konfirmasi → train → test
      [2] Gunakan data lama → langsung train → test
      [0] Keluar
    """
    setup_logging()
    print_header()

    choice = print_menu()

    if choice == "0":
        print("  [OK] Keluar. Sampai jumpa!\n")
        sys.exit(0)

    elif choice == "1":
        # Upload file baru
        result = upload_and_preprocess()
        if result is None:
            print("  [!] Proses dibatalkan.\n")
            sys.exit(0)

        train_csv, test_csv = result

        # Konfirmasi training
        if not confirm_training():
            print("  [OK] Training tidak dijalankan. Data sudah tersimpan di processed/.\n")
            sys.exit(0)

        run_training(train_csv, test_csv)

    elif choice == "2":
        # Gunakan data lama
        existing = check_existing_data()
        if existing is None:
            print("  [!] Data processed tidak ditemukan!")
            print(f"      Pastikan file berikut ada di {config.PROCESSED_DATA_DIR}/:")
            print(f"        - {PROCESSED_TRAIN_NAME}")
            print(f"        - {PROCESSED_TEST_NAME}")
            print("  [!] Silakan pilih opsi [1] untuk upload file baru.\n")
            sys.exit(1)

        train_csv, test_csv = existing
        print(f"  [OK] Data ditemukan:")
        print(f"        Train: {train_csv}")
        print(f"        Test : {test_csv}")

        run_training(train_csv, test_csv)

    print(f"\n{'='*60}")
    print(f"  PIPELINE SELESAI!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
