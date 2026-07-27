"""
preprocessing/word_segmenter.py — Tahap 1: Word Segmentation & Boundary Splitting
==================================================================================
Modul ini bertanggung jawab memisahkan kata-kata yang menempel tanpa spasi,
termasuk nomor urut daftar, batas huruf-angka, CamelCase, dan gabungan kata.

Teknologi/Algoritma:
  - symspellpy  : Word segmentation via Viterbi Algorithm & Dynamic Programming
  - Python re   : Regex Boundary Matching untuk pemisahan pola struktural

Contoh transformasi:
  "1)tidak..."     → "1 tidak..."
  "error500"       → "error 500"
  "gagalSimpan"    → "gagal Simpan"
  "ituerror"       → "itu error"
"""

import re
import logging
from symspellpy import SymSpell

import config

logger = logging.getLogger(__name__)


class WordSegmenter:
    """
    Melakukan pemisahan kata yang menempel berdasarkan pola regex
    dan word segmentation berbasis frequency dictionary (symspellpy).
    """

    def __init__(self, freq_dict_path: str = None):
        """
        Inisialisasi WordSegmenter.

        Args:
            freq_dict_path: Path ke file frequency dictionary untuk symspellpy.
                            Default menggunakan path dari config.py.
        """
        self.freq_dict_path = freq_dict_path or config.FREQ_DICT_PATH
        self._sym_spell = None  # Lazy loading
        logger.info("WordSegmenter diinisialisasi.")

    def _load_symspell(self) -> SymSpell:
        """
        Lazy loading untuk SymSpell instance.
        Hanya dimuat saat pertama kali dibutuhkan untuk menghemat memori.

        Returns:
            SymSpell instance yang sudah dimuat dengan frequency dictionary.
        """
        if self._sym_spell is None:
            logger.info("Memuat frequency dictionary dari: %s", self.freq_dict_path)
            self._sym_spell = SymSpell(
                max_dictionary_edit_distance=config.SYMSPELL_MAX_EDIT_DISTANCE,
                prefix_length=config.SYMSPELL_PREFIX_LENGTH,
            )
            loaded = self._sym_spell.load_dictionary(
                self.freq_dict_path,
                term_index=0,
                count_index=1,
            )
            if not loaded:
                logger.warning(
                    "Gagal memuat frequency dictionary! "
                    "Word segmentation untuk merged words akan dilewati."
                )
            else:
                logger.info("Frequency dictionary berhasil dimuat.")
        return self._sym_spell

    # ------------------------------------------------------------------
    # Sub-tahap 1a: Pisahkan nomor urut daftar yang menempel pada teks
    # ------------------------------------------------------------------
    @staticmethod
    def split_numbered_list(text: str) -> str:
        """
        Memisahkan nomor urut daftar yang menempel pada teks laporan.

        Pattern: digit(s) diikuti ')' lalu langsung huruf tanpa spasi.
        Contoh: "1)tidak bisa" → "1 tidak bisa"
                "2)errorSimpan" → "2 errorSimpan"

        Args:
            text: Teks input yang mungkin berisi nomor daftar menempel.

        Returns:
            Teks dengan nomor daftar yang sudah dipisahkan.
        """
        # Cocokkan: satu/lebih digit + ')' + langsung huruf (tanpa spasi)
        result = re.sub(r"(\d+)\)\s*([a-zA-Z])", r"\1 \2", text)
        return result

    # ------------------------------------------------------------------
    # Sub-tahap 1b: Pisahkan batas antara huruf dan angka
    # ------------------------------------------------------------------
    @staticmethod
    def split_alpha_numeric(text: str) -> str:
        """
        Memisahkan batas antara karakter huruf dan angka yang menempel.

        Contoh: "error500"   → "error 500"
                "http200ok"  → "http 200 ok"
                "v2beta"     → "v 2 beta"

        Args:
            text: Teks input dengan huruf-angka yang menempel.

        Returns:
            Teks dengan batas huruf-angka yang sudah dipisahkan.
        """
        # Huruf diikuti angka: "error500" → "error 500"
        text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
        # Angka diikuti huruf: "500error" → "500 error"
        text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
        return text

    # ------------------------------------------------------------------
    # Sub-tahap 1c: Pisahkan kata berformat CamelCase
    # ------------------------------------------------------------------
    @staticmethod
    def split_camel_case(text: str) -> str:
        """
        Memisahkan kata berformat CamelCase menjadi kata terpisah.

        Contoh: "gagalSimpan"  → "gagal Simpan"
                "sistemCrash"  → "sistem Crash"
                "errorServer"  → "error Server"

        Args:
            text: Teks input yang mungkin mengandung CamelCase.

        Returns:
            Teks dengan CamelCase yang sudah dipisahkan spasi.
        """
        # Sisipkan spasi sebelum huruf kapital yang didahului huruf kecil
        # Pattern: huruf_kecil diikuti huruf_KAPITAL
        result = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        return result

    # ------------------------------------------------------------------
    # Sub-tahap 1d: Word segmentation untuk kata menempel tanpa pola
    # ------------------------------------------------------------------
    def segment_merged_words(self, text: str) -> str:
        """
        Memisahkan gabungan kata yang menempel tanpa spasi menggunakan
        symspellpy word segmentation (Viterbi Algorithm).

        Proses: tokenisasi → segmentasi per token panjang → rekonstruksi.
        Hanya memproses token yang cukup panjang (>7 karakter) dan
        terindikasi sebagai gabungan kata (tidak ditemukan di dictionary).

        Contoh: "ituerror"        → "itu error"
                "tidakbisalogin"  → "tidak bisa login"

        Args:
            text: Teks input yang mungkin berisi kata-kata menempel.

        Returns:
            Teks dengan kata menempel yang sudah dipisahkan.
        """
        sym_spell = self._load_symspell()
        if sym_spell is None:
            return text

        tokens = text.split()
        segmented_tokens = []

        for token in tokens:
            # Hanya proses token alfabet murni yang panjangnya > 7 karakter
            # Token pendek kemungkinan besar kata tunggal, bukan gabungan
            if token.isalpha() and len(token) > 7:
                result = sym_spell.word_segmentation(token.lower())
                # Gunakan hasil segmentasi jika menghasilkan > 1 kata
                if result and " " in result.corrected_string:
                    segmented_tokens.append(result.corrected_string)
                    logger.debug("Segmentasi: '%s' → '%s'", token, result.corrected_string)
                else:
                    segmented_tokens.append(token)
            else:
                segmented_tokens.append(token)

        return " ".join(segmented_tokens)

    # ------------------------------------------------------------------
    # Method utama: jalankan seluruh sub-tahap Word Segmentation
    # ------------------------------------------------------------------
    def process(self, text: str) -> str:
        """
        Menjalankan seluruh sub-tahap Word Segmentation secara berurutan:
        1a → 1b → 1c → 1d

        Args:
            text: Teks mentah input.

        Returns:
            Teks yang sudah melalui seluruh proses word segmentation.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self.split_numbered_list(text)
        text = self.split_alpha_numeric(text)
        text = self.split_camel_case(text)
        text = self.segment_merged_words(text)

        logger.debug("Word Segmentation selesai: '%s'", text[:80])
        return text
