"""
trainer.py — Model Training & Testing Module
==============================================
Modul untuk melatih dan menguji model classifier severity.

Model yang dilatih:
  A. IndoBERT Fine-Tuned (Semantic)
  B. TF-IDF + LinearSVC (N-Gram)
  C. TF-IDF + XGBoost (N-Gram)
  Ensemble: Soft Voting dari ketiga model
"""

import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score

from xgboost import XGBClassifier

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

import config

logger = logging.getLogger(__name__)


# ======================================================================
# PyTorch Dataset
# ======================================================================
class SeverityDataset(Dataset):
    """PyTorch Dataset untuk IndoBERT fine-tuning."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ======================================================================
# Trainer
# ======================================================================
class ModelTrainer:
    """
    Melatih 3 model classifier + ensemble, lalu menyimpan ke disk.
    """

    def __init__(self, train_csv: str, test_csv: str):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model artifacts
        self.le = LabelEncoder()
        self.tfidf = None
        self.model_bert = None
        self.model_svc = None
        self.model_xgb = None
        self.tokenizer = None
        self.num_classes = 0

        # Metrics
        self.metrics = {}

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    def load_data(self):
        """Load train dan test CSV yang sudah dipreprocess.

        Train: wajib memiliki kolom Kategori (dropna diterapkan).
        Test : Kategori bersifat opsional — jika kosong/tidak ada,
               pipeline beralih ke mode prediksi (has_test_labels=False).
        """
        print(f"\n  Memuat data train: {self.train_csv}")
        self.df_train = pd.read_csv(self.train_csv, encoding="utf-8")
        self.df_train = self.df_train.dropna(
            subset=[config.OUTPUT_COLUMN, config.LABEL_COLUMN]
        ).reset_index(drop=True)

        print(f"  Memuat data test : {self.test_csv}")
        self.df_test = pd.read_csv(self.test_csv, encoding="utf-8")
        # Untuk test: hanya buang baris yang tidak punya teks — label opsional
        self.df_test = self.df_test.dropna(
            subset=[config.OUTPUT_COLUMN]
        ).reset_index(drop=True)

        # Deteksi apakah test set memiliki label yang valid
        has_label_col = config.LABEL_COLUMN in self.df_test.columns
        if has_label_col:
            valid_labels = self.df_test[config.LABEL_COLUMN].dropna()
            self.has_test_labels = len(valid_labels) > 0
        else:
            self.has_test_labels = False

        # Encode labels hanya dari train (test mungkin tidak punya label)
        self.le.fit(self.df_train[config.LABEL_COLUMN])
        self.num_classes = len(self.le.classes_)

        self.y_train = self.le.transform(self.df_train[config.LABEL_COLUMN])
        self.X_train_text = self.df_train[config.OUTPUT_COLUMN].values
        self.X_test_text = self.df_test[config.OUTPUT_COLUMN].values

        if self.has_test_labels:
            self.y_test = self.le.transform(self.df_test[config.LABEL_COLUMN])
        else:
            self.y_test = None

        print(f"\n  Train: {len(self.X_train_text)} baris")
        print(f"  Test : {len(self.X_test_text)} baris")
        print(f"  Kelas: {list(self.le.classes_)} ({self.num_classes} kelas)")
        print(f"  Device: {self.device}")
        print(f"  Mode  : {'Evaluasi (label tersedia)' if self.has_test_labels else 'Prediksi (tanpa label)'}")

        print(f"\n  Distribusi label (Train):")
        for i, cls in enumerate(self.le.classes_):
            print(f"    {cls}: {(self.y_train == i).sum()}")

        if self.has_test_labels:
            print(f"  Distribusi label (Test):")
            for i, cls in enumerate(self.le.classes_):
                print(f"    {cls}: {(self.y_test == i).sum()}")
        else:
            print(f"  Test labels: tidak tersedia — mode prediksi aktif")

    # ------------------------------------------------------------------
    # Model A: IndoBERT Fine-Tuned
    # ------------------------------------------------------------------
    def train_indobert(self, epochs=4, batch_size=16, lr=2e-5):
        """Fine-tune IndoBERT dengan classification head."""
        print(f"\n{'='*60}")
        print(f"  MODEL A: IndoBERT Fine-Tuned Classifier")
        print(f"{'='*60}")

        model_name = config.INDOBERT_MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_ds = SeverityDataset(
            self.X_train_text, self.y_train, self.tokenizer, config.INDOBERT_MAX_LENGTH
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        self.model_bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=self.num_classes
        )
        self.model_bert.to(self.device)

        optimizer = torch.optim.AdamW(
            self.model_bert.parameters(), lr=lr, weight_decay=0.01
        )
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        # Training loop
        for epoch in range(epochs):
            self.model_bert.train()
            epoch_loss = 0
            correct = 0
            total = 0

            progress = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs}")
            for batch in progress:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model_bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model_bert.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                progress.set_postfix(
                    loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}"
                )

            print(
                f"  Epoch {epoch+1} — Loss: {epoch_loss/len(train_loader):.4f}, "
                f"Acc: {correct/total:.4f}"
            )

        print("  [OK] IndoBERT training selesai.\n")

    # ------------------------------------------------------------------
    # Model B: TF-IDF + LinearSVC
    # ------------------------------------------------------------------
    def train_linearsvc(self):
        """Train TF-IDF + LinearSVC (dengan CalibratedCV untuk probabilitas)."""
        print(f"\n{'='*60}")
        print(f"  MODEL B: TF-IDF + LinearSVC")
        print(f"{'='*60}")

        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2), max_features=10000, min_df=2,
            max_df=0.95, sublinear_tf=True,
        )
        X_train_tfidf = self.tfidf.fit_transform(self.X_train_text)
        print(f"  TF-IDF features: {X_train_tfidf.shape}")

        base_svc = LinearSVC(
            class_weight="balanced", max_iter=10000, random_state=42
        )
        self.model_svc = CalibratedClassifierCV(base_svc, cv=5)
        self.model_svc.fit(X_train_tfidf, self.y_train)

        print("  [OK] LinearSVC training selesai.\n")

    # ------------------------------------------------------------------
    # Model C: TF-IDF + XGBoost
    # ------------------------------------------------------------------
    def train_xgboost(self):
        """Train XGBoost menggunakan TF-IDF features yang sama."""
        print(f"\n{'='*60}")
        print(f"  MODEL C: TF-IDF + XGBoost")
        print(f"{'='*60}")

        X_train_tfidf = self.tfidf.transform(self.X_train_text)

        self.model_xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
        )
        self.model_xgb.fit(X_train_tfidf, self.y_train, verbose=False)

        print("  [OK] XGBoost training selesai.\n")

    # ------------------------------------------------------------------
    # Shared: run inference on X_test_text, return ensemble predictions
    # ------------------------------------------------------------------
    def _run_inference(self):
        """
        Jalankan inferensi semua model pada self.X_test_text.

        Returns:
            tuple: (all_probs_bert, probs_svc, probs_xgb,
                    all_preds_bert, preds_svc, preds_xgb,
                    ensemble_preds, ensemble_probs, W_BERT, W_SVC, W_XGB)
        """
        X_test_tfidf = self.tfidf.transform(self.X_test_text)

        # --- IndoBERT ---
        self.model_bert.eval()
        # SeverityDataset memerlukan label; gunakan placeholder nol
        dummy_labels = np.zeros(len(self.X_test_text), dtype=int)
        test_ds = SeverityDataset(
            self.X_test_text, dummy_labels, self.tokenizer, config.INDOBERT_MAX_LENGTH
        )
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        all_preds_bert = []
        all_probs_bert = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="  IndoBERT Inferensi"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model_bert(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                probs = torch.softmax(outputs.logits, dim=1)
                all_preds_bert.extend(torch.argmax(probs, dim=1).cpu().numpy())
                all_probs_bert.extend(probs.cpu().numpy())

        all_preds_bert = np.array(all_preds_bert)
        all_probs_bert = np.array(all_probs_bert)

        # --- LinearSVC ---
        preds_svc = self.model_svc.predict(X_test_tfidf)
        probs_svc = self.model_svc.predict_proba(X_test_tfidf)

        # --- XGBoost ---
        preds_xgb = self.model_xgb.predict(X_test_tfidf)
        probs_xgb = self.model_xgb.predict_proba(X_test_tfidf)

        # --- Ensemble (Soft Voting) ---
        W_BERT, W_SVC, W_XGB = 0.5, 0.25, 0.25
        ensemble_probs = (
            W_BERT * all_probs_bert + W_SVC * probs_svc + W_XGB * probs_xgb
        )
        ensemble_preds = np.argmax(ensemble_probs, axis=1)

        return (
            all_probs_bert, probs_svc, probs_xgb,
            all_preds_bert, preds_svc, preds_xgb,
            ensemble_preds, ensemble_probs,
            W_BERT, W_SVC, W_XGB,
        )

    # ------------------------------------------------------------------
    # Evaluate (mode: label tersedia)
    # ------------------------------------------------------------------
    def test_all_models(self):
        """Evaluasi semua model pada test set dan tampilkan perbandingan.

        Hanya dipanggil ketika self.has_test_labels == True.
        """
        print(f"\n{'='*60}")
        print(f"  EVALUASI PADA TEST SET")
        print(f"{'='*60}")

        (
            all_probs_bert, probs_svc, probs_xgb,
            all_preds_bert, preds_svc, preds_xgb,
            ensemble_preds, ensemble_probs,
            W_BERT, W_SVC, W_XGB,
        ) = self._run_inference()

        # --- Metrics ---
        models = {
            "IndoBERT Fine-Tuned": all_preds_bert,
            "TF-IDF + LinearSVC": preds_svc,
            "TF-IDF + XGBoost": preds_xgb,
            "Ensemble (Soft Voting)": ensemble_preds,
        }

        print(f"\n{'='*60}")
        print(f"  HASIL EVALUASI")
        print(f"{'='*60}")

        for name, preds in models.items():
            acc = accuracy_score(self.y_test, preds)
            f1 = f1_score(self.y_test, preds, average="weighted")
            self.metrics[name] = {"accuracy": float(acc), "f1": float(f1)}

            print(f"\n  --- {name} ---")
            print(f"  Accuracy : {acc:.4f}")
            print(f"  F1 Score : {f1:.4f}")
            print(classification_report(
                self.y_test, preds, target_names=self.le.classes_, zero_division=0
            ))

        # Comparison table
        print(f"\n{'='*60}")
        print(f"  PERBANDINGAN MODEL")
        print(f"{'='*60}")
        print(f"  {'Model':<30s} {'Accuracy':>10s} {'F1 Score':>10s}")
        print(f"  {'-'*50}")
        best_name, best_f1 = "", 0
        for name, m in self.metrics.items():
            print(f"  {name:<30s} {m['accuracy']:>10.4f} {m['f1']:>10.4f}")
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_name = name
        print(f"  {'-'*50}")
        print(f"  Best: {best_name} (F1: {best_f1:.4f})")
        print(f"{'='*60}")

        # Store ensemble weights for save_models()
        self._ensemble_weights = {"indobert": W_BERT, "linearsvc": W_SVC, "xgboost": W_XGB}

    # ------------------------------------------------------------------
    # Predict only (mode: tanpa label)
    # ------------------------------------------------------------------
    def predict_only(self, output_dir: str = None):
        """Prediksi Kategori untuk setiap baris test set tanpa label.

        Menggunakan ensemble soft-voting dari ketiga model terlatih.
        Hasil disimpan HANYA ke data/output/test_prediction.csv.
        File data/processed/test_cleaned.csv tidak dimodifikasi.

        Args:
            output_dir: Direktori output. Default: config.OUTPUT_DATA_DIR.
        """
        print(f"\n{'='*60}")
        print(f"  PREDIKSI TEST SET (mode: tanpa label)")
        print(f"{'='*60}")

        (
            _, _, _,
            _, _, _,
            ensemble_preds, _,
            W_BERT, W_SVC, W_XGB,
        ) = self._run_inference()

        # Decode prediksi ke nama kelas
        predicted_labels = self.le.inverse_transform(ensemble_preds)

        # Susun DataFrame output — pertahankan urutan baris asli
        result_df = self.df_test.copy()
        result_df[config.LABEL_COLUMN] = predicted_labels

        # Kolom output: Log Temuan, Kategori (prediksi), cleaned_text
        cols_order = []
        if config.INPUT_COLUMN in result_df.columns:
            cols_order.append(config.INPUT_COLUMN)
        cols_order.append(config.LABEL_COLUMN)
        if config.OUTPUT_COLUMN in result_df.columns:
            cols_order.append(config.OUTPUT_COLUMN)
        # Tambahkan kolom lain yang mungkin ada
        for col in result_df.columns:
            if col not in cols_order:
                cols_order.append(col)
        result_df = result_df[cols_order]

        # Tentukan path output
        if output_dir is None:
            output_dir = os.path.join(config.BASE_DIR, "data", "output")
        os.makedirs(output_dir, exist_ok=True)
        prediction_path = os.path.join(output_dir, "test_prediction.csv")

        result_df.to_csv(prediction_path, index=False, encoding="utf-8")

        # Juga timpa test_cleaned.csv agar downstream tetap konsisten
        processed_test_path = os.path.join(config.PROCESSED_DATA_DIR, "test_cleaned.csv")
        result_df.to_csv(processed_test_path, index=False, encoding="utf-8")

        # Ringkasan
        print(f"\n  Prediksi selesai: {len(result_df)} baris")
        print(f"  Distribusi prediksi:")
        for cls, count in result_df[config.LABEL_COLUMN].value_counts().items():
            print(f"    {cls}: {count}")
        print(f"\n  [OK] Hasil disimpan ke  : {prediction_path}")
        print(f"  [OK] test_cleaned.csv diperbarui: {processed_test_path}")
        print(f"{'='*60}")

        # Store ensemble weights for save_models()
        self._ensemble_weights = {"indobert": W_BERT, "linearsvc": W_SVC, "xgboost": W_XGB}

    # ------------------------------------------------------------------
    # Save Models
    # ------------------------------------------------------------------
    def save_models(self):
        """Simpan semua model ke direktori models/."""
        os.makedirs(config.MODELS_DIR, exist_ok=True)

        print(f"\n  Menyimpan model ke: {config.MODELS_DIR}/")

        # 1. IndoBERT
        bert_path = os.path.join(config.MODELS_DIR, "indobert_severity.pt")
        torch.save(
            {
                "model_state_dict": self.model_bert.state_dict(),
                "model_name": config.INDOBERT_MODEL_NAME,
                "num_classes": self.num_classes,
                "max_length": config.INDOBERT_MAX_LENGTH,
                "label_classes": list(self.le.classes_),
            },
            bert_path,
        )
        print(f"  [OK] {bert_path}")

        # 2. TF-IDF Vectorizer
        tfidf_path = os.path.join(config.MODELS_DIR, "tfidf_vectorizer.pkl")
        with open(tfidf_path, "wb") as f:
            pickle.dump(self.tfidf, f)
        print(f"  [OK] {tfidf_path}")

        # 3. LinearSVC
        svc_path = os.path.join(config.MODELS_DIR, "linearsvc_model.pkl")
        with open(svc_path, "wb") as f:
            pickle.dump(self.model_svc, f)
        print(f"  [OK] {svc_path}")

        # 4. XGBoost
        xgb_path = os.path.join(config.MODELS_DIR, "xgboost_model.pkl")
        with open(xgb_path, "wb") as f:
            pickle.dump(self.model_xgb, f)
        print(f"  [OK] {xgb_path}")

        # 5. Label Encoder
        le_path = os.path.join(config.MODELS_DIR, "label_encoder.pkl")
        with open(le_path, "wb") as f:
            pickle.dump(self.le, f)
        print(f"  [OK] {le_path}")

        # 6. Ensemble Config
        ensemble_cfg = {
            "weights": self._ensemble_weights,
            "num_classes": self.num_classes,
            "label_classes": list(self.le.classes_),
            "metrics": self.metrics,
        }
        cfg_path = os.path.join(config.MODELS_DIR, "ensemble_config.json")
        with open(cfg_path, "w") as f:
            json.dump(ensemble_cfg, f, indent=2)
        print(f"  [OK] {cfg_path}")

        # List files
        print(f"\n  Model files:")
        for fname in sorted(os.listdir(config.MODELS_DIR)):
            fsize = os.path.getsize(os.path.join(config.MODELS_DIR, fname))
            print(f"    {fname:40s} {fsize/1024/1024:.1f} MB")

    # ------------------------------------------------------------------
    # Run All
    # ------------------------------------------------------------------
    def run(self):
        """Jalankan full pipeline: load → train → evaluasi/prediksi → save.

        Otomatis memilih mode:
          - Evaluasi : jika test set memiliki label valid (has_test_labels=True)
          - Prediksi : jika test set tidak memiliki label (has_test_labels=False)
        """
        self.load_data()
        self.train_indobert()
        self.train_linearsvc()
        self.train_xgboost()

        if self.has_test_labels:
            self.test_all_models()
        else:
            self.predict_only()

        self.save_models()
        print(f"\n  [OK] Semua model berhasil dilatih dan disimpan!\n")
