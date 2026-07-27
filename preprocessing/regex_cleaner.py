"""
preprocessing/regex_cleaner.py — Tahap 3: Regex Cleaning
=========================================================
Modul ini menghapus derau (noise) dari teks menggunakan pola regex,
termasuk tag HTML, URL, simbol/karakter khusus, dan spasi berlebih.

Teknologi/Algoritma:
  - Python re (NFA/DFA Pattern Matching Engine)
  - Compiled regex patterns untuk performa optimal

Contoh transformasi:
  "<b>error</b>"              → "error"
  "cek https://simata.dev/bug" → "cek"
  "error!@# pada sistem"       → "error pada sistem"
"""

import re
import logging

logger = logging.getLogger(__name__)


class RegexCleaner:
    """
    Membersihkan teks dari noise menggunakan regex patterns.
    Patterns dikompilasi saat inisialisasi untuk performa optimal
    pada pemrosesan batch (compiled once, used many).
    """

    def __init__(self):
        """
        Inisialisasi RegexCleaner dengan compiled regex patterns.
        Pre-compilation menghindari overhead kompilasi berulang
        saat memproses banyak dokumen.
        """
        # Pattern 1: Tag HTML (termasuk self-closing tags)
        self._html_pattern = re.compile(r"<[^>]+>", re.IGNORECASE)

        # Pattern 2: URL (http, https, ftp, dan www)
        self._url_pattern = re.compile(
            r"https?://\S+|www\.\S+|ftp://\S+", re.IGNORECASE
        )

        # Pattern 3: Karakter non-alfanumerik dan non-spasi
        # Mempertahankan hanya huruf kecil, angka, dan spasi
        self._special_chars_pattern = re.compile(r"[^a-z0-9\s]")

        # Pattern 4: Spasi berlebih (2+ spasi berturut-turut)
        self._whitespace_pattern = re.compile(r"\s+")

        logger.info("RegexCleaner diinisialisasi dengan 4 compiled patterns.")

    def remove_html_tags(self, text: str) -> str:
        """
        Menghapus seluruh tag HTML dari teks.

        Pattern `<[^>]+>` mencocokkan tag pembuka, penutup,
        dan self-closing (e.g., <br/>, <div class="x">).

        Contoh: "<b>error</b> pada <div>form</div>" → "error pada form"

        Args:
            text: Teks yang mungkin mengandung tag HTML.

        Returns:
            Teks bersih tanpa tag HTML.
        """
        result = self._html_pattern.sub(" ", text)
        return result

    def remove_urls(self, text: str) -> str:
        """
        Menghapus seluruh URL dari teks.

        Menangani URL dengan protokol http/https/ftp dan URL
        yang diawali dengan www.

        Contoh: "cek https://simata.dev/bug/123" → "cek"

        Args:
            text: Teks yang mungkin mengandung URL.

        Returns:
            Teks bersih tanpa URL.
        """
        result = self._url_pattern.sub(" ", text)
        return result

    def remove_special_chars(self, text: str) -> str:
        """
        Menghapus simbol dan karakter khusus yang tidak relevan.

        Hanya mempertahankan huruf kecil (a-z), angka (0-9),
        dan spasi. Karakter lain dihapus.

        CATATAN: Tahap ini dijalankan SETELAH case folding,
        sehingga hanya perlu mempertahankan huruf kecil.

        Contoh: "error!@# pada sistem..." → "error pada sistem"

        Args:
            text: Teks yang mungkin mengandung karakter khusus.

        Returns:
            Teks bersih hanya berisi alfanumerik dan spasi.
        """
        result = self._special_chars_pattern.sub(" ", text)
        return result

    def normalize_whitespace(self, text: str) -> str:
        """
        Menormalisasi spasi berlebih menjadi satu spasi,
        serta menghapus spasi di awal dan akhir teks.

        Contoh: "error   pada   sistem  " → "error pada sistem"

        Args:
            text: Teks dengan kemungkinan spasi berlebih.

        Returns:
            Teks dengan spasi yang sudah dinormalisasi.
        """
        result = self._whitespace_pattern.sub(" ", text).strip()
        return result

    def process(self, text: str) -> str:
        """
        Menjalankan seluruh sub-tahap Regex Cleaning secara berurutan:
        HTML tags → URLs → Special chars → Whitespace normalization

        Args:
            text: Teks input yang perlu dibersihkan.

        Returns:
            Teks bersih dari noise.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)

        logger.debug("Regex Cleaning selesai: '%s'", text[:80])
        return text
