"""
preprocessing/feature_extractor.py — Tahap 2: Feature Extraction & Embedding
==============================================================================
Modul ini menggunakan IndoBERT (indobenchmark/indobert-base-p1) untuk
menghasilkan embedding vektor dari teks yang sudah dibersihkan.

Teknologi/Algoritma:
  - HuggingFace Transformers — Model loading & inference
  - IndoBERT WordPiece Tokenizer — Tokenisasi sub-kata
  - IndoBERT Encoder — Contextual embedding (768-dim)

Keunggulan IndoBERT dibanding pipeline tradisional:
  - Slang/singkatan (blm, tdk, yg) di-handle secara kontekstual
    melalui WordPiece tokenization + attention mechanism
  - Stopwords tidak perlu dihapus karena model memahami konteks
  - HTML token [HTML_TAG] bisa menjadi fitur tambahan

Contoh:
  Input : "data tidak muncul setelah disimpan"
  Output: numpy array shape (768,) — [CLS] embedding
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import untuk menghindari overhead saat modul tidak digunakan
_transformers_available = None


def _check_transformers():
    """Cek apakah transformers dan torch tersedia."""
    global _transformers_available
    if _transformers_available is None:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            _transformers_available = True
        except ImportError:
            _transformers_available = False
    return _transformers_available


class FeatureExtractor:
    """
    Mengekstrak fitur dari teks menggunakan IndoBERT encoder.
    Menghasilkan embedding vektor 768 dimensi per teks menggunakan
    representasi [CLS] token dari lapisan terakhir.
    """

    def __init__(self, model_name: str = None, max_length: int = None):
        """
        Inisialisasi FeatureExtractor.

        Args:
            model_name: Nama model HuggingFace. Default dari config.
            max_length: Panjang maksimum token. Default dari config.
        """
        import config
        self.model_name = model_name or config.INDOBERT_MODEL_NAME
        self.max_length = max_length or config.INDOBERT_MAX_LENGTH

        self._tokenizer = None
        self._model = None
        self._device = None

        logger.info(
            "FeatureExtractor diinisialisasi — model: %s, max_length: %d",
            self.model_name, self.max_length,
        )

    def _load_model(self):
        """
        Lazy loading untuk tokenizer dan model IndoBERT.
        Model hanya dimuat saat pertama kali dibutuhkan.
        Otomatis mendeteksi GPU (CUDA) jika tersedia.
        """
        if self._model is not None:
            return

        if not _check_transformers():
            raise ImportError(
                "Library 'transformers' dan 'torch' dibutuhkan untuk FeatureExtractor. "
                "Install dengan: pip install transformers torch"
            )

        import torch
        from transformers import AutoTokenizer, AutoModel

        logger.info("Memuat model IndoBERT: %s ...", self.model_name)

        # Deteksi device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            logger.info("GPU terdeteksi — menggunakan CUDA.")
        else:
            self._device = torch.device("cpu")
            logger.info("GPU tidak terdeteksi — menggunakan CPU.")

        # Load tokenizer dan model
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()  # Set ke evaluation mode (no dropout)

        logger.info("Model IndoBERT berhasil dimuat.")

    def tokenize(self, text: str) -> dict:
        """
        Tokenisasi teks menggunakan IndoBERT WordPiece Tokenizer.

        Args:
            text: Teks bersih hasil Text Cleaning.

        Returns:
            Dictionary berisi input_ids, attention_mask, dll.
        """
        self._load_model()
        return self._tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def extract_embedding(self, text: str) -> np.ndarray:
        """
        Mengekstrak embedding vektor [CLS] dari teks.

        Proses:
        1. Tokenisasi teks via WordPiece
        2. Forward pass melalui IndoBERT
        3. Ambil representasi [CLS] token (index 0) dari lapisan terakhir
        4. Konversi ke numpy array

        Args:
            text: Teks bersih hasil Text Cleaning.

        Returns:
            numpy array shape (768,) — embedding vektor.
        """
        import torch

        self._load_model()

        if not text or not text.strip():
            return np.zeros(768)

        # Tokenisasi
        inputs = self._tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Pindahkan ke device yang tepat
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Forward pass tanpa gradient (inference only)
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Ambil [CLS] token embedding (index 0 dari last_hidden_state)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()

        # Konversi ke numpy array di CPU
        return cls_embedding.cpu().numpy()

    def extract_embeddings_batch(self, texts: list, batch_size: int = 16) -> np.ndarray:
        """
        Mengekstrak embedding untuk batch teks sekaligus.
        Lebih efisien daripada memanggil extract_embedding() per teks.

        Args:
            texts:      List teks bersih.
            batch_size: Jumlah teks per batch. Default 16.

        Returns:
            numpy array shape (n_texts, 768).
        """
        import torch

        self._load_model()

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Ganti teks kosong dengan placeholder
            batch_texts = [t if t and t.strip() else "[PAD]" for t in batch_texts]

            inputs = self._tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

            logger.debug(
                "Batch %d/%d selesai (%d teks).",
                i // batch_size + 1,
                (len(texts) + batch_size - 1) // batch_size,
                len(batch_texts),
            )

        return np.vstack(all_embeddings)

    def process(self, text: str) -> str:
        """
        Interface konsisten dengan tahap lain.
        Mengembalikan teks apa adanya (embedding diekstrak terpisah
        via extract_embedding atau extract_embeddings_batch).

        Tahap ini tidak mengubah teks — hanya mengekstrak fitur.
        Untuk penggunaan di pipeline, gunakan extract_embedding() langsung.

        Args:
            text: Teks input.

        Returns:
            Teks input tanpa perubahan.
        """
        return text if isinstance(text, str) else ""
