import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
evaluate_api.py — Evaluasi Model via API
==========================================
Mengirim setiap Log Temuan dari test.csv ke API /api/predict,
lalu membandingkan hasil prediksi ensemble dengan label sebenarnya.

Output:
  - Accuracy, Weighted F1, Macro F1
  - Classification Report per kelas
  - Confusion Matrix
  - Detail per-row (prediksi vs aktual)
  - Export ke data/output/evaluation_results.csv
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config

# ======================================================================
# Configuration
# ======================================================================
API_URL = "http://127.0.0.1:8000/api/predict"
TEST_CSV = os.path.join(config.BASE_DIR, "test.csv")
OUTPUT_DIR = os.path.join(config.DATA_DIR, "output")
RESULT_CSV = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
LABEL_ORDER = ["Fatal", "Mayor", "Minor", "Kosmetik"]


def main():
    # ------------------------------------------------------------------
    # 1. Load test data
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  EVALUASI MODEL VIA API")
    print(f"{'='*60}")

    df = pd.read_csv(TEST_CSV, encoding="utf-8")
    df = df.dropna(subset=["Log Temuan", "Kategori"]).reset_index(drop=True)

    print(f"  Test data : {TEST_CSV}")
    print(f"  Total rows: {len(df)}")
    print(f"  Distribusi label aktual:")
    for cat in LABEL_ORDER:
        count = (df["Kategori"] == cat).sum()
        print(f"    {cat:10s}: {count}")

    # ------------------------------------------------------------------
    # 2. Health check
    # ------------------------------------------------------------------
    try:
        health = requests.get("http://127.0.0.1:8000/api/health", timeout=5)
        health_data = health.json()
        if not health_data.get("models_loaded"):
            print("  [!] Model belum dimuat. Tunggu server selesai loading.")
            sys.exit(1)
        print(f"\n  [OK] API connected — device: {health_data.get('device')}")
    except Exception as e:
        print(f"  [!] Tidak bisa terhubung ke API: {e}")
        print(f"      Pastikan server berjalan: python -m uvicorn api_server:app --port 8000")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Predict all rows
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  MENGIRIM PREDIKSI KE API ({len(df)} rows)...")
    print(f"{'='*60}")

    predictions = []
    confidences = []
    all_probs = []
    errors = []

    start_time = time.time()

    for idx, row in df.iterrows():
        text = str(row["Log Temuan"])
        try:
            resp = requests.post(
                API_URL,
                json={"text": text},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            predictions.append(data["prediction"])
            confidences.append(data["confidence"])
            all_probs.append(data["probabilities"])

            status = "OK" if data["prediction"] == row["Kategori"] else "MISS"
            print(
                f"  [{idx+1:3d}/{len(df)}] {status:4s}  "
                f"Actual: {row['Kategori']:10s} | "
                f"Pred: {data['prediction']:10s} | "
                f"Conf: {data['confidence']:.3f}"
            )
        except Exception as e:
            predictions.append("ERROR")
            confidences.append(0.0)
            all_probs.append({})
            errors.append((idx, str(e)))
            print(f"  [{idx+1:3d}/{len(df)}] ERR   ERROR: {e}")

    elapsed = time.time() - start_time
    print(f"\n  Waktu total: {elapsed:.1f}s ({elapsed/len(df):.2f}s per row)")

    # ------------------------------------------------------------------
    # 4. Build result DataFrame
    # ------------------------------------------------------------------
    df["Prediksi"] = predictions
    df["Confidence"] = confidences
    df["Benar"] = df["Kategori"] == df["Prediksi"]

    # Add per-class probabilities
    for cat in LABEL_ORDER:
        df[f"Prob_{cat}"] = [p.get(cat, 0.0) for p in all_probs]

    # Filter out errors for metrics
    df_valid = df[df["Prediksi"] != "ERROR"].copy()
    y_true = df_valid["Kategori"].values
    y_pred = df_valid["Prediksi"].values

    if len(errors) > 0:
        print(f"\n  [!] {len(errors)} row gagal diprediksi (dikeluarkan dari metrik).")

    # ------------------------------------------------------------------
    # 5. Compute metrics
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  HASIL EVALUASI ENSEMBLE (Soft-Voting)")
    print(f"{'='*60}")

    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision (weighted): {precision_w:.4f}")
    print(f"  Recall (weighted)   : {recall_w:.4f}")
    print(f"  F1 Score (weighted) : {f1_weighted:.4f}")
    print(f"  F1 Score (macro)    : {f1_macro:.4f}")
    print(f"  Total benar         : {int(df_valid['Benar'].sum())} / {len(df_valid)}")

    # --- Classification Report ---
    print(f"\n{'='*60}")
    print(f"  CLASSIFICATION REPORT")
    print(f"{'='*60}")
    report = classification_report(
        y_true, y_pred,
        labels=LABEL_ORDER,
        target_names=LABEL_ORDER,
        zero_division=0,
    )
    print(report)

    # --- Confusion Matrix ---
    print(f"{'='*60}")
    print(f"  CONFUSION MATRIX")
    print(f"{'='*60}")
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    # Print header
    print(f"\n  {'Actual \\ Pred':<14s}", end="")
    for cat in LABEL_ORDER:
        print(f"  {cat:>10s}", end="")
    print()
    print(f"  {'-'*54}")

    for i, cat in enumerate(LABEL_ORDER):
        print(f"  {cat:<14s}", end="")
        for j in range(len(LABEL_ORDER)):
            val = cm[i][j]
            marker = " <" if i == j else ""
            print(f"  {val:>10d}", end="")
        print()

    # --- Per-class detail ---
    print(f"\n{'='*60}")
    print(f"  ANALISIS PER KELAS")
    print(f"{'='*60}")

    per_class_report = classification_report(
        y_true, y_pred,
        labels=LABEL_ORDER,
        target_names=LABEL_ORDER,
        zero_division=0,
        output_dict=True,
    )

    for cat in LABEL_ORDER:
        metrics = per_class_report[cat]
        actual_count = int(metrics["support"])
        predicted_count = int((y_pred == cat).sum())
        correct_count = int(cm[LABEL_ORDER.index(cat)][LABEL_ORDER.index(cat)])

        print(f"\n  [{cat}]")
        print(f"    Jumlah aktual  : {actual_count}")
        print(f"    Jumlah prediksi: {predicted_count}")
        print(f"    Benar          : {correct_count}")
        print(f"    Precision      : {metrics['precision']:.4f}")
        print(f"    Recall         : {metrics['recall']:.4f}")
        print(f"    F1-Score       : {metrics['f1-score']:.4f}")

    # --- Misclassified rows ---
    misclassified = df_valid[~df_valid["Benar"]].copy()
    if len(misclassified) > 0:
        print(f"\n{'='*60}")
        print(f"  DETAIL KESALAHAN KLASIFIKASI ({len(misclassified)} rows)")
        print(f"{'='*60}")
        for _, row in misclassified.iterrows():
            text_preview = str(row["Log Temuan"])[:80].replace("\n", " ")
            print(
                f"\n  Row {row.name+1}:"
                f"\n    Actual : {row['Kategori']}"
                f"\n    Pred   : {row['Prediksi']} ({row['Confidence']:.3f})"
                f"\n    Text   : {text_preview}..."
            )

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(RESULT_CSV, index=False, encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"  HASIL DISIMPAN")
    print(f"{'='*60}")
    print(f"  [OK] {RESULT_CSV}")

    # --- Summary metrics to JSON ---
    import json
    summary = {
        "accuracy": round(acc, 4),
        "precision_weighted": round(precision_w, 4),
        "recall_weighted": round(recall_w, 4),
        "f1_weighted": round(f1_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "total_samples": len(df_valid),
        "correct": int(df_valid["Benar"].sum()),
        "incorrect": int((~df_valid["Benar"]).sum()),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "elapsed_seconds": round(elapsed, 2),
    }
    for cat in LABEL_ORDER:
        m = per_class_report[cat]
        summary["per_class"][cat] = {
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1_score": round(m["f1-score"], 4),
            "support": int(m["support"]),
        }

    json_path = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {json_path}")

    print(f"\n{'='*60}")
    print(f"  EVALUASI SELESAI!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
