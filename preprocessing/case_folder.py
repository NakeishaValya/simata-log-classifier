"""
preprocessing/case_folder.py — Tahap 2: Case Folding (Lowercasing)
==================================================================
Modul ini mengubah seluruh karakter teks menjadi huruf kecil (lowercase)
secara seragam untuk menghilangkan variasi kapitalisasi.

Teknologi/Algoritma:
  - Python Built-in str.lower() berbasis Unicode/ASCII Case Mapping Algorithm

Contoh transformasi:
  "Itu Error"    → "itu error"
  "GAGAL SIMPAN" → "gagal simpan"
  "Error500"     → "error500"
"""

import logging

logger = logging.getLogger(__name__)


class CaseFolder:
    """
    Melakukan case folding (konversi ke lowercase) pada teks.
    Tahap ini penting untuk memastikan konsistensi representasi
    kata sebelum proses normalisasi dan filtering selanjutnya.
    """

    def __init__(self):
        """Inisialisasi CaseFolder."""
        logger.info("CaseFolder diinisialisasi.")

    @staticmethod
    def fold_case(text: str) -> str:
        """
        Mengubah seluruh karakter string menjadi huruf kecil.

        Menggunakan str.lower() yang mendukung Unicode Case Mapping,
        sehingga karakter non-ASCII juga ditangani dengan benar.

        Args:
            text: Teks input dengan variasi kapitalisasi.

        Returns:
            Teks yang sudah dikonversi ke lowercase seluruhnya.
        """
        return text.lower()

    def process(self, text: str) -> str:
        """
        Method utama: menjalankan case folding.

        Args:
            text: Teks input.

        Returns:
            Teks lowercase.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        result = self.fold_case(text)
        logger.debug("Case Folding selesai: '%s'", result[:80])
        return result
