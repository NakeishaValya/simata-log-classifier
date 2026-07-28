"""
preprocessing/text_cleaner.py — Tahap 1: Text Cleaning & HTML Preservation
===========================================================================
Modul ini menggabungkan seluruh pembersihan teks berbasis Regex Engine
menjadi satu pipeline yang koheren. Menangani:

  1. Preservasi tag HTML → token [HTML_TAG]
  2. Pembersihan URL
  3. Pemisahan nomor urut daftar (1), 2.) dari teks
  4. Pemisahan CamelCase (gagalSimpan → gagal Simpan)
  5. Pemisahan batas huruf-angka (error500 → error 500)
  6. Penghapusan kode/ID komponen angka panjang (≥ 5 digit)
  7. Penghapusan karakter khusus non-alfanumerik
  8. Lowercasing (Case Folding)
  9. Normalisasi spasi berlebih

Teknologi/Algoritma:
  - Python re (Compiled Regex Patterns — NFA/DFA Engine)
  - Compiled once at __init__, reused per-row → performa batch optimal

Contoh transformasi end-to-end:
  Input : "<b>1)Error500</b> komponen 12345 gagalSimpan https://x.co"
  Output: "[HTML_TAG] error [HTML_TAG] komponen gagal simpan"
"""

import re
import logging

from preprocessing.slang_normalizer import SlangNormalizer

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Membersihkan teks mentah dari noise menggunakan regex patterns.
    Semua patterns dikompilasi saat inisialisasi untuk performa optimal
    pada pemrosesan batch (compiled once, used many).
    """

    def __init__(self):
        """
        Inisialisasi TextCleaner dengan compiled regex patterns.
        Pre-compilation menghindari overhead kompilasi berulang
        saat memproses ribuan baris dokumen.
        """
        # --- Compiled Patterns ---

        # 1. Tag HTML (termasuk self-closing tags dan HTML entities)
        self._html_tag_pattern = re.compile(r"<[^>]+>", re.IGNORECASE)
        self._html_entity_pattern = re.compile(r"&\w+;")

        # 2. URL (http, https, ftp, www)
        self._url_pattern = re.compile(
            r"https?://\S+|www\.\S+|ftp://\S+", re.IGNORECASE
        )

        # 3. Nomor urut daftar: "1)" atau "2." yang menempel pada huruf
        self._numbered_list_pattern = re.compile(
            r"(\d+)[)\.](\s*)([a-zA-Z])"
        )

        # 4. CamelCase: huruf kecil diikuti huruf kapital
        self._camel_case_pattern = re.compile(r"([a-z])([A-Z])")

        # 5. Batas huruf-angka
        self._alpha_to_num_pattern = re.compile(r"([a-zA-Z])(\d)")
        self._num_to_alpha_pattern = re.compile(r"(\d)([a-zA-Z])")

        # 6. Kode/ID angka panjang (≥ 5 digit) — dihapus seluruhnya
        self._long_numeric_pattern = re.compile(r"\b\d{5,}\b")

        # 7. Karakter non-alfanumerik dan non-spasi (selain [HTML_TAG] token)
        #    Dijalankan SETELAH case folding, jadi hanya perlu a-z
        self._special_chars_pattern = re.compile(r"[^a-z0-9\s\[\]_]")

        # 8. Spasi berlebih (2+ spasi berturut-turut)
        self._whitespace_pattern = re.compile(r"\s+")
        self._slang_normalizer = SlangNormalizer()

        logger.info(
            "TextCleaner diinisialisasi dengan %d compiled regex patterns.",
            8,
        )

    # ------------------------------------------------------------------
    # Sub-tahap 1: Preservasi HTML Tags → [HTML_TAG]
    # ------------------------------------------------------------------
    def preserve_html_tags(self, text: str) -> str:
        """
        Mengubah tag HTML dan HTML entities menjadi token [HTML_TAG].

        Tag HTML dipreservasi sebagai token khusus agar informasi
        struktural tetap tersedia untuk IndoBERT di tahap berikutnya.

        Contoh: "<b>error</b> &amp; crash" → "[HTML_TAG] error [HTML_TAG] [HTML_TAG] crash"

        Args:
            text: Teks yang mungkin mengandung tag HTML.

        Returns:
            Teks dengan tag HTML diganti token [HTML_TAG].
        """
        text = self._html_tag_pattern.sub(" [HTML_TAG] ", text)
        text = self._html_entity_pattern.sub(" [HTML_TAG] ", text)
        return text

    # ------------------------------------------------------------------
    # Sub-tahap 2: Hapus URL
    # ------------------------------------------------------------------
    def remove_urls(self, text: str) -> str:
        """
        Menghapus seluruh URL dari teks.

        Contoh: "cek https://simata.dev/bug/123" → "cek"

        Args:
            text: Teks yang mungkin mengandung URL.

        Returns:
            Teks bersih tanpa URL.
        """
        return self._url_pattern.sub(" ", text)

    # ------------------------------------------------------------------
    # Sub-tahap 3: Pisahkan nomor urut daftar
    # ------------------------------------------------------------------
    def split_numbered_list(self, text: str) -> str:
        """
        Memisahkan nomor urut daftar yang menempel pada teks.

        Contoh: "1)tidak bisa" → "1 tidak bisa"
                "2.errorSimpan" → "2 errorSimpan"

        Args:
            text: Teks input.

        Returns:
            Teks dengan nomor daftar yang sudah dipisahkan.
        """
        return self._numbered_list_pattern.sub(r"\1 \3", text)

    # ------------------------------------------------------------------
    # Sub-tahap 4: Pisahkan CamelCase
    # ------------------------------------------------------------------
    def split_camel_case(self, text: str) -> str:
        """
        Memisahkan kata berformat CamelCase menjadi kata terpisah.

        Contoh: "gagalSimpan" → "gagal Simpan"

        Args:
            text: Teks input.

        Returns:
            Teks dengan CamelCase yang sudah dipisahkan.
        """
        return self._camel_case_pattern.sub(r"\1 \2", text)

    # ------------------------------------------------------------------
    # Sub-tahap 5: Pisahkan batas huruf-angka
    # ------------------------------------------------------------------
    def split_alpha_numeric(self, text: str) -> str:
        """
        Memisahkan batas antara karakter huruf dan angka yang menempel.

        Contoh: "error500" → "error 500"
                "500error" → "500 error"

        Args:
            text: Teks input.

        Returns:
            Teks dengan batas huruf-angka yang sudah dipisahkan.
        """
        text = self._alpha_to_num_pattern.sub(r"\1 \2", text)
        text = self._num_to_alpha_pattern.sub(r"\1 \2", text)
        return text

    # ------------------------------------------------------------------
    # Sub-tahap 6: Hapus kode/ID angka panjang (≥ 5 digit)
    # ------------------------------------------------------------------
    def remove_long_numeric_ids(self, text: str) -> str:
        """
        Menghapus token berupa angka panjang (≥ 5 digit) yang biasanya
        merupakan kode komponen, ID tiket, atau nomor referensi.

        Contoh: "komponen 12345 error" → "komponen error"

        Args:
            text: Teks input.

        Returns:
            Teks tanpa kode angka panjang.
        """
        return self._long_numeric_pattern.sub(" ", text)

    # ------------------------------------------------------------------
    # Sub-tahap 7: Hapus karakter khusus
    # ------------------------------------------------------------------
    def remove_special_chars(self, text: str) -> str:
        """
        Menghapus simbol dan karakter khusus yang tidak relevan.
        Mempertahankan huruf kecil (a-z), angka (0-9), spasi,
        dan karakter [ ] _ (untuk token [HTML_TAG]).

        CATATAN: Dijalankan SETELAH case folding.

        Args:
            text: Teks input.

        Returns:
            Teks bersih.
        """
        return self._special_chars_pattern.sub(" ", text)

    # ------------------------------------------------------------------
    # Sub-tahap 8: Case Folding (Lowercasing)
    # ------------------------------------------------------------------
    @staticmethod
    def fold_case(text: str) -> str:
        """
        Mengubah seluruh karakter menjadi huruf kecil.

        Args:
            text: Teks input.

        Returns:
            Teks lowercase.
        """
        return text.lower()

    # ------------------------------------------------------------------
    # Sub-tahap 9: Normalisasi spasi
    # ------------------------------------------------------------------
    def normalize_whitespace(self, text: str) -> str:
        """
        Menormalisasi spasi berlebih menjadi satu spasi
        dan menghapus spasi di awal/akhir teks.

        Args:
            text: Teks input.

        Returns:
            Teks dengan spasi yang dinormalisasi.
        """
        return self._whitespace_pattern.sub(" ", text).strip()

    # ------------------------------------------------------------------
    # Method utama: jalankan seluruh sub-tahap berurutan
    # ------------------------------------------------------------------
    def process(self, text: str) -> str:
        """
        Menjalankan seluruh sub-tahap Text Cleaning secara berurutan:
        HTML Preservation → URL → List Numbers → CamelCase →
        Alpha-Numeric → Long IDs → Case Folding → Special Chars →
        Slang Normalization → Whitespace Normalization

        Args:
            text: Teks mentah input.

        Returns:
            Teks bersih hasil cleaning, siap untuk IndoBERT.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self.preserve_html_tags(text)
        text = self.remove_urls(text)
        text = self.split_numbered_list(text)
        text = self.split_camel_case(text)
        text = self.split_alpha_numeric(text)
        text = self.remove_long_numeric_ids(text)
        text = self.fold_case(text)
        text = self.remove_special_chars(text)
        text = self._slang_normalizer.process(text)
        text = self.normalize_whitespace(text)

        logger.debug("Text Cleaning selesai: '%s'", text[:80])
        return text
