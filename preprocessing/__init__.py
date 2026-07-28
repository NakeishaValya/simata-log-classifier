"""
preprocessing/__init__.py — Package Init & Public API
======================================================
Mengekspor seluruh komponen preprocessing agar bisa diimpor
langsung dari package `preprocessing`.

Pipeline 2 Tahap:
  1. TextCleaner         — Text Cleaning & HTML Preservation
  2. FeatureExtractor    — IndoBERT Feature Extraction & Embedding

Contoh penggunaan:
    from preprocessing import PreprocessingPipeline
    pipeline = PreprocessingPipeline()
    cleaned = pipeline.preprocess_text("<b>1)Error500</b> gagalSimpan")
"""

from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.text_cleaner import TextCleaner
from preprocessing.slang_normalizer import SlangNormalizer
from preprocessing.feature_extractor import FeatureExtractor

__all__ = [
    "PreprocessingPipeline",
    "TextCleaner",
    "SlangNormalizer",
    "FeatureExtractor",
]
