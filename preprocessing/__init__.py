"""
preprocessing/__init__.py — Package Init & Public API
======================================================
Mengekspor seluruh komponen preprocessing agar bisa diimpor
langsung dari package `preprocessing`.

Contoh penggunaan:
    from preprocessing import PreprocessingPipeline
    pipeline = PreprocessingPipeline()
    result = pipeline.preprocess_text("1)tidak bisa login, error500")
"""

from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.word_segmenter import WordSegmenter
from preprocessing.case_folder import CaseFolder
from preprocessing.regex_cleaner import RegexCleaner
from preprocessing.slang_normalizer import SlangNormalizer
from preprocessing.stopword_filter import StopwordFilter

__all__ = [
    "PreprocessingPipeline",
    "WordSegmenter",
    "CaseFolder",
    "RegexCleaner",
    "SlangNormalizer",
    "StopwordFilter",
]
