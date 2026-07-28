"""
preprocessing/slang_normalizer.py — Normalisasi Slang
=====================================================
Menormalisasi kata slang/singkatan ke bentuk baku Bahasa Indonesia.

Pendekatan:
  - Dictionary lookup dari CSV lokal
  - Tokenisasi dengan nlp_id bila tersedia
  - Fallback aman ke split biasa jika dependency tidak bisa dipakai
"""

import logging
import os
import re
import urllib.parse

import pandas as pd

import config

try:
    from nlp_id.tokenizer import Tokenizer as NlpIdTokenizer
except Exception:  # pragma: no cover
    NlpIdTokenizer = None

logger = logging.getLogger(__name__)


class SlangNormalizer:
    """Normalisasi slang berbasis kamus."""

    def __init__(self, slang_dict_path: str = None):
        default_path = getattr(
            config,
            "SLANG_DICT_PATH",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dictionaries", "kamus_slang.csv"),
        )
        self.slang_dict_path = slang_dict_path or default_path
        self._slang_dict = None
        self._tokenizer = self._init_tokenizer()
        logger.info("SlangNormalizer diinisialisasi dengan sumber: %s", self.slang_dict_path)

    def _is_url(self, path: str) -> bool:
        parsed = urllib.parse.urlparse(path)
        return parsed.scheme in ("http", "https")

    def _init_tokenizer(self):
        if NlpIdTokenizer is None:
            logger.warning(
                "nlp_id tidak tersedia. Slang normalizer menggunakan tokenisasi split biasa."
            )
            return None
        try:
            tokenizer = NlpIdTokenizer()
            logger.info("Tokenizer nlp_id berhasil diinisialisasi.")
            return tokenizer
        except Exception as exc:
            logger.warning("Gagal inisialisasi tokenizer nlp_id (%s). Fallback ke split biasa.", exc)
            return None

    def _load_slang_dict(self) -> dict:
        if self._slang_dict is None:
            logger.info("Memuat kamus slang dari: %s", self.slang_dict_path)
            try:
                df = pd.read_csv(self.slang_dict_path, encoding="utf-8")

                if not {"slang", "baku"}.issubset(df.columns):
                    raise ValueError("CSV kamus slang harus memiliki kolom 'slang' dan 'baku'")

                self._slang_dict = dict(
                    zip(
                        df["slang"].astype(str).str.strip().str.lower(),
                        df["baku"].astype(str).str.strip().str.lower(),
                    )
                )
                logger.info("Kamus slang berhasil dimuat: %d entri.", len(self._slang_dict))
            except FileNotFoundError:
                logger.error("File kamus slang tidak ditemukan: %s", self.slang_dict_path)
                self._slang_dict = {}
            except Exception as exc:
                logger.error("Gagal memuat kamus slang: %s", exc)
                self._slang_dict = {}
        return self._slang_dict

    def _tokenize(self, text: str):
        if self._tokenizer is not None:
            try:
                return self._tokenizer.tokenize(text)
            except Exception as exc:
                logger.warning("Tokenisasi nlp_id gagal (%s). Fallback ke split biasa.", exc)
        return text.split()

    @staticmethod
    def _detokenize(tokens: list) -> str:
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([([{])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]}])", r"\1", text)
        return text

    def normalize_slang(self, text: str) -> str:
        slang_dict = self._load_slang_dict()
        if not slang_dict:
            return text

        tokens = self._tokenize(text)
        normalized_tokens = []
        for token in tokens:
            normalized_tokens.append(slang_dict.get(token.lower(), token))
        return self._detokenize(normalized_tokens)

    def get_dict_size(self) -> int:
        return len(self._load_slang_dict())

    def process(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        return self.normalize_slang(text)
