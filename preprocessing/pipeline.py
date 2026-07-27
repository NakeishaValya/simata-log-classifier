"""
preprocessing/pipeline.py — Pipeline Orchestrator
===================================================
Modul orchestrator yang menjalankan seluruh 5 tahap preprocessing
secara berurutan pada satu teks atau seluruh DataFrame sekaligus.

Pipeline Flow:
  Input Text (raw)
      │
      ├─ Tahap 1: Word Segmentation & Boundary Splitting
      ├─ Tahap 2: Case Folding (Lowercasing)
      ├─ Tahap 3: Regex Cleaning
      ├─ Tahap 4: Dynamic Slang Normalization
      └─ Tahap 5: Filtering & Custom Stopwords
      │
  Output Text (clean)
"""

import logging
import pandas as pd
from tqdm import tqdm

import config
from preprocessing.word_segmenter import WordSegmenter
from preprocessing.case_folder import CaseFolder
from preprocessing.regex_cleaner import RegexCleaner
from preprocessing.slang_normalizer import SlangNormalizer
from preprocessing.stopword_filter import StopwordFilter

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Orchestrator untuk 5 tahap preprocessing teks.
    Menginisialisasi seluruh modul dan menjalankannya secara berurutan.
    Mendukung pemrosesan teks tunggal maupun batch (DataFrame).
    """

    def __init__(
        self,
        freq_dict_path: str = None,
        slang_dict_path: str = None,
        custom_whitelist: set = None,
    ):
        """
        Inisialisasi pipeline dengan seluruh modul preprocessing.

        Args:
            freq_dict_path:   Path frequency dictionary untuk word segmentation.
            slang_dict_path:  Path kamus slang untuk normalisasi.
            custom_whitelist: Set kata tambahan yang harus dipertahankan
                              oleh stopword filter.
        """
        logger.info("=" * 60)
        logger.info("Inisialisasi PreprocessingPipeline...")
        logger.info("=" * 60)

        # Inisialisasi seluruh modul preprocessing
        self.word_segmenter = WordSegmenter(freq_dict_path=freq_dict_path)
        self.case_folder = CaseFolder()
        self.regex_cleaner = RegexCleaner()
        self.slang_normalizer = SlangNormalizer(slang_dict_path=slang_dict_path)
        self.stopword_filter = StopwordFilter(custom_whitelist=custom_whitelist)

        # Daftar tahap pipeline beserta toggle dari config
        self._stages = [
            ("word_segmentation", "Tahap 1: Word Segmentation", self.word_segmenter),
            ("case_folding", "Tahap 2: Case Folding", self.case_folder),
            ("regex_cleaning", "Tahap 3: Regex Cleaning", self.regex_cleaner),
            ("slang_normalization", "Tahap 4: Slang Normalization", self.slang_normalizer),
            ("stopword_filtering", "Tahap 5: Stopword Filtering", self.stopword_filter),
        ]

        logger.info("Pipeline siap dengan %d tahap aktif.", self._count_active_stages())

    def _count_active_stages(self) -> int:
        """Menghitung jumlah tahap yang aktif berdasarkan config."""
        return sum(
            1 for key, _, _ in self._stages
            if config.PIPELINE_STAGES.get(key, True)
        )

    def preprocess_text(self, text: str, verbose: bool = False) -> str:
        """
        Memproses satu teks melalui seluruh tahap pipeline.

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

        current_text = text

        for stage_key, stage_name, stage_module in self._stages:
            # Cek apakah tahap ini diaktifkan di config
            if not config.PIPELINE_STAGES.get(stage_key, True):
                logger.info("SKIP  : %s (dinonaktifkan di config)", stage_name)
                if verbose:
                    print(f"  SKIP  : {stage_name}")
                continue

            # Jalankan tahap preprocessing
            current_text = stage_module.process(current_text)

            if verbose:
                print(f"  {stage_name}")
                print(f"    -> {current_text}")

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

        Menambahkan kolom baru berisi teks yang sudah diproses.
        Menampilkan progress bar via tqdm untuk monitoring batch.

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

        # Proses setiap baris dengan progress bar
        tqdm.pandas(desc="Preprocessing")
        result_df[output_col] = result_df[input_col].progress_apply(
            lambda text: self.preprocess_text(
                str(text) if pd.notna(text) else "", verbose=verbose
            )
        )

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
        Shortcut: baca CSV → preprocess → simpan CSV.

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
            # Pastikan direktori output ada
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            logger.info("Hasil disimpan ke: %s", output_path)

        return result_df
