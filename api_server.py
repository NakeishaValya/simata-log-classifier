"""
api_server.py — FastAPI Prediction Server untuk SIMATA Bug Classifier
======================================================================
Server API yang memuat artefak model Soft-Voting Ensemble (IndoBERT +
LinearSVC + XGBoost) dan menyediakan endpoint prediksi severity secara
real-time.

Cara pakai:
  $ python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  GET  /              → Halaman web form (static file)
  GET  /api/health    → Health check
  POST /api/predict   → Prediksi severity dari teks Log Temuan
"""

import json
import logging
import os
import pickle
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from preprocessing.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)

# ======================================================================
# Pydantic Models
# ======================================================================

class PredictRequest(BaseModel):
    """Request body untuk endpoint /api/predict."""
    text: str = Field(
        ...,
        min_length=1,
        description="Teks Log Temuan yang akan diprediksi severity-nya.",
        examples=[
            "Saat pertama kali membuka Modul Preview Dokumen, "
            "sistem melakukan rollback otomatis namun data induk ikut terhapus permanen"
        ],
    )


class PredictResponse(BaseModel):
    """Response body dari endpoint /api/predict."""
    prediction: str = Field(..., description="Kategori severity hasil prediksi ensemble.")
    confidence: float = Field(..., description="Skor confidence (probabilitas tertinggi).")
    probabilities: dict = Field(
        ...,
        description="Probabilitas per kelas severity.",
    )
    cleaned_text: str = Field(..., description="Teks setelah preprocessing (text cleaning).")


# ======================================================================
# Global Model Container
# ======================================================================
class ModelContainer:
    """
    Menyimpan seluruh artefak model yang sudah dimuat ke memori.
    Load once at startup, reuse for every prediction request.
    """

    def __init__(self):
        self.label_encoder = None
        self.tfidf_vectorizer = None
        self.model_svc = None
        self.model_xgb = None
        self.model_bert = None
        self.tokenizer_bert = None
        self.ensemble_weights = {}
        self.label_classes = []
        self.num_classes = 0
        self.device = None
        self.text_cleaner = None
        self.is_loaded = False

    def load(self):
        """Memuat semua artefak model dari direktori models/."""
        models_dir = config.MODELS_DIR
        logger.info("Memuat artefak model dari: %s", models_dir)

        # Device detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", self.device)

        # 1. Label Encoder
        le_path = os.path.join(models_dir, "label_encoder.pkl")
        with open(le_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        logger.info("[OK] Label Encoder dimuat — kelas: %s", list(self.label_encoder.classes_))

        # 2. TF-IDF Vectorizer
        tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        with open(tfidf_path, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)
        logger.info("[OK] TF-IDF Vectorizer dimuat.")

        # 3. LinearSVC (CalibratedClassifierCV)
        svc_path = os.path.join(models_dir, "linearsvc_model.pkl")
        with open(svc_path, "rb") as f:
            self.model_svc = pickle.load(f)
        logger.info("[OK] LinearSVC dimuat.")

        # 4. XGBoost
        xgb_path = os.path.join(models_dir, "xgboost_model.pkl")
        with open(xgb_path, "rb") as f:
            self.model_xgb = pickle.load(f)
        logger.info("[OK] XGBoost dimuat.")

        # 5. IndoBERT
        bert_path = os.path.join(models_dir, "indobert_severity.pt")
        checkpoint = torch.load(bert_path, map_location=self.device, weights_only=False)

        model_name = checkpoint.get("model_name", config.INDOBERT_MODEL_NAME)
        num_classes = checkpoint.get("num_classes", 4)
        self.num_classes = num_classes

        self.tokenizer_bert = AutoTokenizer.from_pretrained(model_name)
        self.model_bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )
        self.model_bert.load_state_dict(checkpoint["model_state_dict"])
        self.model_bert.to(self.device)
        self.model_bert.eval()
        logger.info("[OK] IndoBERT dimuat — model: %s, classes: %d", model_name, num_classes)

        # 6. Ensemble Config
        cfg_path = os.path.join(models_dir, "ensemble_config.json")
        with open(cfg_path, "r") as f:
            ensemble_cfg = json.load(f)
        self.ensemble_weights = ensemble_cfg.get("weights", {
            "indobert": 0.5, "linearsvc": 0.25, "xgboost": 0.25
        })
        self.label_classes = ensemble_cfg.get(
            "label_classes", list(self.label_encoder.classes_)
        )
        logger.info(
            "[OK] Ensemble Config dimuat — weights: %s, classes: %s",
            self.ensemble_weights, self.label_classes,
        )

        # 7. Text Cleaner
        self.text_cleaner = TextCleaner()
        logger.info("[OK] TextCleaner diinisialisasi.")

        self.is_loaded = True
        logger.info("=== Semua model berhasil dimuat! Server siap menerima request. ===")

    def predict(self, raw_text: str) -> dict:
        """
        Menjalankan full prediction pipeline pada satu teks.

        Args:
            raw_text: Teks mentah dari Log Temuan.

        Returns:
            dict dengan keys: prediction, confidence, probabilities, cleaned_text
        """
        if not self.is_loaded:
            raise RuntimeError("Model belum dimuat!")

        # --- Step 1: Text Cleaning ---
        cleaned_text = self.text_cleaner.process(raw_text)
        if not cleaned_text.strip():
            return {
                "prediction": self.label_classes[0] if self.label_classes else "Unknown",
                "confidence": 0.0,
                "probabilities": {cls: 0.0 for cls in self.label_classes},
                "cleaned_text": "",
            }

        # --- Step 2: TF-IDF Models ---
        tfidf_features = self.tfidf_vectorizer.transform([cleaned_text])
        probs_svc = self.model_svc.predict_proba(tfidf_features)  # shape: (1, n_classes)
        probs_xgb = self.model_xgb.predict_proba(tfidf_features)  # shape: (1, n_classes)

        # --- Step 3: IndoBERT ---
        inputs = self.tokenizer_bert(
            cleaned_text,
            max_length=config.INDOBERT_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model_bert(**inputs)
        probs_bert = torch.softmax(outputs.logits, dim=1).cpu().numpy()  # shape: (1, n_classes)

        # --- Step 4: Soft-Voting Ensemble ---
        w_bert = self.ensemble_weights.get("indobert", 0.5)
        w_svc = self.ensemble_weights.get("linearsvc", 0.25)
        w_xgb = self.ensemble_weights.get("xgboost", 0.25)

        ensemble_probs = (
            w_bert * probs_bert + w_svc * probs_svc + w_xgb * probs_xgb
        )[0]  # shape: (n_classes,)

        # --- Step 5: Decode ---
        predicted_idx = int(np.argmax(ensemble_probs))
        predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(ensemble_probs[predicted_idx])

        probabilities = {}
        for idx, cls in enumerate(self.label_classes):
            probabilities[cls] = round(float(ensemble_probs[idx]), 4)

        return {
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "cleaned_text": cleaned_text,
        }


# ======================================================================
# Singleton Model Container
# ======================================================================
model_container = ModelContainer()


# ======================================================================
# FastAPI Lifespan (model loading at startup)
# ======================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models saat server startup, cleanup saat shutdown."""
    # --- Startup ---
    log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )
    logger.info("Server dimulai — memuat model...")
    try:
        model_container.load()
    except Exception as e:
        logger.error("GAGAL memuat model: %s", e)
        raise

    yield

    # --- Shutdown ---
    logger.info("Server shutdown.")


# ======================================================================
# FastAPI Application
# ======================================================================
app = FastAPI(
    title="SIMATA Bug Classifier API",
    description="API untuk klasifikasi severity bug menggunakan Soft-Voting Ensemble "
                "(IndoBERT + LinearSVC + XGBoost).",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for web UI
WEB_DIR = os.path.join(config.BASE_DIR, "web")
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ======================================================================
# Routes
# ======================================================================
@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve halaman utama (Form Tambah Log Temuan)."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "SIMATA Bug Classifier API is running. Web UI not found."}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": model_container.is_loaded,
        "device": str(model_container.device) if model_container.device else "not initialized",
        "label_classes": model_container.label_classes,
        "ensemble_weights": model_container.ensemble_weights,
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict_severity(request: PredictRequest):
    """
    Prediksi severity dari teks Log Temuan.

    Proses:
      1. Text Cleaning (regex + slang normalization)
      2. TF-IDF → LinearSVC + XGBoost predict_proba
      3. IndoBERT → softmax probabilities
      4. Soft-Voting Ensemble (weighted average)
      5. Return prediksi + confidence + probabilities
    """
    if not model_container.is_loaded:
        raise HTTPException(status_code=503, detail="Model belum dimuat. Server sedang loading.")

    try:
        result = model_container.predict(request.text)
        return PredictResponse(**result)
    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
