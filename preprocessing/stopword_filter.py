"""
preprocessing/stopword_filter.py — Tahap 5: Filtering & Custom Stopwords
==========================================================================
Modul ini membuang kata hubung umum (stopwords) yang tidak bermakna,
NAMUN mempertahankan kata-kata negasi dan indikator keparahan error
melalui mekanisme Exception Logic / Whitelist.

Teknologi/Algoritma:
  - PySastrawi (StopWordRemoverFactory) — Source stopwords Bahasa Indonesia
  - Set Membership Filtering (Hash Set Lookup) — O(1) per lookup

Kata-kata yang DIPERTAHANKAN (whitelist):
  Negasi   : "tidak", "belum", "gagal", "bukan", "jangan", "tanpa", ...
  Severity : "error", "crash", "hang", "fatal", "bug", "null", ...

Contoh transformasi:
  "data yang sudah di hapus itu tidak muncul"
  → "data hapus tidak muncul"
  (stopwords "yang", "sudah", "di", "itu" dihapus;
   "tidak" DIPERTAHANKAN karena ada di whitelist)
"""

import logging
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import config

logger = logging.getLogger(__name__)


class StopwordFilter:
    """
    Filter stopwords dengan Exception Logic / Whitelist.
    Menggunakan daftar stopwords dari PySastrawi sebagai basis,
    lalu menerapkan whitelist agar kata-kata penting tidak terhapus.
    """

    def __init__(self, custom_whitelist: set = None):
        """
        Inisialisasi StopwordFilter.

        Args:
            custom_whitelist: Set kata-kata tambahan yang harus dipertahankan.
                              Akan digabung dengan whitelist default dari config.
                              Default: None (hanya menggunakan whitelist config).
        """
        # Muat stopwords dari PySastrawi
        factory = StopWordRemoverFactory()
        self._base_stopwords = set(factory.get_stop_words())

        # Gabungkan whitelist default dari config dengan custom whitelist
        self._whitelist = set(config.WHITELIST_WORDS)
        if custom_whitelist:
            self._whitelist |= custom_whitelist

        # Buat final stopwords: hapus kata whitelist dari daftar stopwords
        # Ini adalah inti dari Exception Logic
        self._effective_stopwords = self._base_stopwords - self._whitelist

        logger.info(
            "StopwordFilter diinisialisasi — "
            "Base stopwords: %d, Whitelist: %d, Effective stopwords: %d",
            len(self._base_stopwords),
            len(self._whitelist),
            len(self._effective_stopwords),
        )

    def get_stopwords(self) -> set:
        """
        Mengembalikan set stopwords efektif (sudah dikurangi whitelist).

        Returns:
            Set kata-kata yang akan dihapus dari teks.
        """
        return self._effective_stopwords.copy()

    def get_whitelist(self) -> set:
        """
        Mengembalikan set kata-kata whitelist yang dipertahankan.

        Returns:
            Set kata-kata yang dilindungi dari penghapusan.
        """
        return self._whitelist.copy()

    def is_whitelisted(self, word: str) -> bool:
        """
        Mengecek apakah suatu kata ada di whitelist.

        Args:
            word: Kata yang akan dicek.

        Returns:
            True jika kata ada di whitelist, False jika tidak.
        """
        return word.lower() in self._whitelist

    def filter_stopwords(self, text: str) -> str:
        """
        Menghapus stopwords dari teks dengan Exception Logic.

        Proses:
        1. Tokenisasi teks berdasarkan spasi
        2. Untuk setiap token, cek apakah termasuk effective stopwords
        3. Jika YA (stopword & BUKAN whitelist) → hapus
        4. Jika TIDAK → pertahankan
        5. Rekonstruksi teks

        Kompleksitas: O(n) dimana n = jumlah token,
        dengan O(1) lookup per token (Hash Set).

        Contoh:
            Input:  "data yang sudah di hapus itu tidak muncul"
            Output: "data hapus tidak muncul"

        Args:
            text: Teks input yang akan difilter.

        Returns:
            Teks bersih tanpa stopwords (kecuali yang di-whitelist).
        """
        tokens = text.split()
        filtered_tokens = []

        for token in tokens:
            if token.lower() in self._effective_stopwords:
                logger.debug("Stopword dihapus: '%s'", token)
            else:
                filtered_tokens.append(token)

        return " ".join(filtered_tokens)

    def process(self, text: str) -> str:
        """
        Method utama: menjalankan stopword filtering.

        Args:
            text: Teks input.

        Returns:
            Teks yang sudah difilter stopwords.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        result = self.filter_stopwords(text)
        logger.debug("Stopword Filtering selesai: '%s'", result[:80])
        return result
