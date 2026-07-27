"""
preprocessing/slang_normalizer.py — Tahap 4: Dynamic Slang Normalization
=========================================================================
Modul ini menormalisasi kata-kata slang, singkatan, dan bahasa tidak baku
yang umum digunakan oleh tester/QA menjadi kata baku Bahasa Indonesia.

Teknologi/Algoritma:
  - Hash Map Lookup (Python dict) — O(1) time complexity per lookup
  - pandas untuk loading kamus dari CSV
  - Pendekatan dictionary-based (bukan hardcoded) agar mudah di-extend

Contoh transformasi:
  "yg"        → "yang"
  "blm"       → "belum"
  "tdk"       → "tidak"
  "strukdat"  → "struktur data"
"""

import logging
import pandas as pd

import config

logger = logging.getLogger(__name__)


class SlangNormalizer:
    """
    Menormalisasi kata slang/singkatan menjadi kata baku menggunakan
    dictionary lookup. Kamus dimuat dari file CSV eksternal sehingga
    bisa diperluas tanpa mengubah kode.
    """

    def __init__(self, slang_dict_path: str = None):
        """
        Inisialisasi SlangNormalizer.

        Args:
            slang_dict_path: Path ke file CSV kamus slang.
                             Format CSV: kolom 'slang' dan 'baku'.
                             Default menggunakan path dari config.py.
        """
        self.slang_dict_path = slang_dict_path or config.SLANG_DICT_PATH
        self._slang_dict = None  # Lazy loading
        logger.info("SlangNormalizer diinisialisasi.")

    def _load_slang_dict(self) -> dict:
        """
        Lazy loading kamus slang dari file CSV ke Python dict (HashMap).

        File CSV harus memiliki dua kolom:
        - 'slang': kata slang/singkatan (key)
        - 'baku' : kata baku pengganti (value)

        Returns:
            Dictionary mapping {slang: baku}.

        Raises:
            FileNotFoundError: Jika file kamus tidak ditemukan.
        """
        if self._slang_dict is None:
            logger.info("Memuat kamus slang dari: %s", self.slang_dict_path)
            try:
                df = pd.read_csv(self.slang_dict_path, encoding="utf-8")
                # Konversi ke dict: key=slang (lowercase), value=baku
                self._slang_dict = dict(
                    zip(
                        df["slang"].str.strip().str.lower(),
                        df["baku"].str.strip().str.lower(),
                    )
                )
                logger.info(
                    "Kamus slang berhasil dimuat: %d entri.", len(self._slang_dict)
                )
            except FileNotFoundError:
                logger.error(
                    "File kamus slang tidak ditemukan: %s", self.slang_dict_path
                )
                self._slang_dict = {}
            except Exception as e:
                logger.error("Gagal memuat kamus slang: %s", str(e))
                self._slang_dict = {}
        return self._slang_dict

    def normalize_slang(self, text: str) -> str:
        """
        Menormalisasi kata slang/singkatan dalam teks menjadi kata baku.

        Proses:
        1. Tokenisasi teks berdasarkan spasi
        2. Lookup setiap token di kamus slang (O(1) per token)
        3. Ganti token yang cocok dengan kata baku
        4. Rekonstruksi teks dari token-token

        Contoh: "yg blm bisa login" → "yang belum bisa login"

        Args:
            text: Teks input yang mungkin mengandung slang.

        Returns:
            Teks dengan slang yang sudah dinormalisasi ke kata baku.
        """
        slang_dict = self._load_slang_dict()
        if not slang_dict:
            return text

        tokens = text.split()
        normalized_tokens = []

        for token in tokens:
            # Lookup token di kamus slang (case-insensitive, sudah lowercase)
            replacement = slang_dict.get(token.lower(), token)
            if replacement != token:
                logger.debug("Normalisasi slang: '%s' → '%s'", token, replacement)
            normalized_tokens.append(replacement)

        return " ".join(normalized_tokens)

    def get_dict_size(self) -> int:
        """
        Mengembalikan jumlah entri dalam kamus slang.

        Returns:
            Jumlah mapping slang → baku yang dimuat.
        """
        return len(self._load_slang_dict())

    def process(self, text: str) -> str:
        """
        Method utama: menjalankan normalisasi slang.

        Args:
            text: Teks input.

        Returns:
            Teks yang sudah dinormalisasi.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        result = self.normalize_slang(text)
        logger.debug("Slang Normalization selesai: '%s'", result[:80])
        return result
