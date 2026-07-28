"""
preprocessing/pipeline.py — Pipeline Orchestrator
===================================================
Modul orchestrator yang menjalankan 2 tahap preprocessing
secara berurutan pada satu teks atau seluruh DataFrame sekaligus.

Pipeline Flow (2 Tahap):
  Input Text (raw)
      │
      ├─ Tahap 1: Text Cleaning & HTML Preservation (Regex Engine)
      │     ├─ Tag HTML → token [HTML_TAG]
      │     ├─ Pisahkan list 1), 2., CamelCase, huruf-angka
      │     ├─ Hapus ID/kode komponen ≥ 5 digit
      │     └─ Lowercasing (Case Folding)
      │
      └─ Tahap 2: Feature Extraction & Embedding (IndoBERT Encoder)
            ├─ WordPiece Tokenizer membaca teks bersih
            ├─ IndoBERT memetakan slang/singkatan ke vektor konteks
            └─ Output: embedding vector 768-dim per teks
      │
  Output: DataFrame (cleaned_text + embeddings)
"""

import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from preprocessing.text_cleaner import TextCleaner
from preprocessing.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Orchestrator untuk 2 tahap preprocessing teks.

    Tahap 1 (Text Cleaning) selalu dijalankan.
    Tahap 2 (Feature Extraction) bersifat opsional — hanya dijalankan
    jika toggle 'feature_extraction' aktif di config.
    """

    def __init__(self):
        """
        Inisialisasi pipeline dengan seluruh modul preprocessing.
        """
        logger.info("=" * 60)
        logger.info("Inisialisasi PreprocessingPipeline (2 tahap)...")
        logger.info("=" * 60)

        # Tahap 1: Text Cleaning (selalu aktif)
        self.text_cleaner = TextCleaner()

        # Tahap 2: Feature Extraction (opsional)
        self.feature_extractor = None
        if config.PIPELINE_STAGES.get("feature_extraction", True):
            try:
                self.feature_extractor = FeatureExtractor()
                logger.info("Tahap 2 (Feature Extraction) aktif.")
            except Exception as e:
                logger.warning(
                    "Gagal menginisialisasi FeatureExtractor: %s. "
                    "Pipeline akan berjalan tanpa embedding.",
                    str(e),
                )
        else:
            logger.info("Tahap 2 (Feature Extraction) dinonaktifkan di config.")

        logger.info("Pipeline siap.")

    def preprocess_text(self, text: str, verbose: bool = False) -> str:
        """
        Memproses satu teks melalui Tahap 1 (Text Cleaning).

        Args:
            text:    Teks mentah input (Log Temuan).
            verbose: Jika True, cetak hasil setiap tahap ke console.

        Returns:
            Teks bersih hasil preprocessing.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        if verbose:
            print(f"\n{'='*60}")
            print(f"INPUT : {text}")
            print(f"{'='*60}")

        # Tahap 1: Text Cleaning
        if config.PIPELINE_STAGES.get("text_cleaning", True):
            current_text = self.text_cleaner.process(text)
            if verbose:
                print(f"  Tahap 1: Text Cleaning & HTML Preservation")
                print(f"    -> {current_text}")
        else:
            current_text = text
            if verbose:
                print(f"  SKIP : Tahap 1 (dinonaktifkan)")

        if verbose:
            print(f"{'='*60}")
            print(f"OUTPUT: {current_text}")
            print(f"{'='*60}\n")

        return current_text

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        input_column: str = None,
        output_column: str = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Memproses seluruh DataFrame melalui pipeline preprocessing.

        Tahap 1: Text Cleaning → kolom cleaned_text
        Tahap 2: Feature Extraction → file embedding .npy terpisah

        Args:
            df:            DataFrame pandas berisi data temuan.
            input_column:  Nama kolom input (default dari config).
            output_column: Nama kolom output (default dari config).
            verbose:       Jika True, cetak detail per baris.

        Returns:
            DataFrame dengan kolom output tambahan berisi teks bersih.
        """
        input_col = input_column or config.INPUT_COLUMN
        output_col = output_column or config.OUTPUT_COLUMN

        if input_col not in df.columns:
            raise ValueError(
                f"Kolom '{input_col}' tidak ditemukan dalam DataFrame. "
                f"Kolom yang tersedia: {list(df.columns)}"
            )

        logger.info(
            "Memproses DataFrame: %d baris, kolom input='%s', output='%s'",
            len(df), input_col, output_col,
        )

        # Buat copy agar tidak mengubah DataFrame asli
        result_df = df.copy()

        # ---- Tahap 1: Text Cleaning ----
        print("\n  [Tahap 1/2] Text Cleaning & HTML Preservation...")
        tqdm.pandas(desc="  Text Cleaning")
        result_df[output_col] = result_df[input_col].progress_apply(
            lambda text: self.preprocess_text(
                str(text) if pd.notna(text) else "", verbose=verbose
            )
        )

        # ---- Tahap 2: Feature Extraction (IndoBERT) ----
        if self.feature_extractor is not None:
            print("\n  [Tahap 2/2] Feature Extraction (IndoBERT)...")
            try:
                texts = result_df[output_col].tolist()
                embeddings = self.feature_extractor.extract_embeddings_batch(texts)
                # Simpan embedding sebagai atribut di DataFrame
                result_df.attrs["embeddings"] = embeddings
                result_df.attrs["embedding_dim"] = embeddings.shape[1]
                logger.info(
                    "Feature Extraction selesai. Shape: %s", embeddings.shape
                )
            except Exception as e:
                logger.error("Feature Extraction gagal: %s", str(e))
                print(f"  [!] Feature Extraction gagal: {e}")
        else:
            print("\n  [Tahap 2/2] Feature Extraction — DILEWATI")

        # Statistik hasil
        empty_count = (result_df[output_col] == "").sum()
        logger.info(
            "Preprocessing selesai. Total: %d baris, kosong: %d baris.",
            len(result_df), empty_count,
        )

        return result_df

    def preprocess_csv(
        self,
        input_path: str,
        output_path: str = None,
        input_column: str = None,
        output_column: str = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Shortcut: baca CSV → preprocess → simpan CSV + embedding .npy.

        Args:
            input_path:    Path file CSV input.
            output_path:   Path file CSV output. None = tidak disimpan.
            input_column:  Nama kolom input.
            output_column: Nama kolom output.
            verbose:       Jika True, cetak detail.

        Returns:
            DataFrame hasil preprocessing.
        """
        logger.info("Membaca CSV dari: %s", input_path)
        df = pd.read_csv(input_path, encoding="utf-8")
        logger.info("CSV berhasil dibaca: %d baris, %d kolom.", len(df), len(df.columns))

        result_df = self.preprocess_dataframe(
            df, input_column, output_column, verbose
        )

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Simpan CSV (teks bersih)
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            logger.info("Hasil teks bersih disimpan ke: %s", output_path)

            # Simpan embedding sebagai .npy jika ada
            if "embeddings" in result_df.attrs:
                npy_path = output_path.replace(".csv", "_embeddings.npy")
                np.save(npy_path, result_df.attrs["embeddings"])
                logger.info("Embedding disimpan ke: %s", npy_path)

        return result_df
