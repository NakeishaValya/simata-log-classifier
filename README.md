# SIMATA Bug Classifier

Pipeline klasifikasi otomatis tingkat keparahan temuan/bug dari teks "Log Temuan" tester pada sistem **Web SIMATA**. Dilengkapi dengan **antarmuka web real-time** untuk rekomendasi severity otomatis.

## Deskripsi

Sistem ini memproses teks mentah dari tester, melatih tiga model klasifikasi secara paralel, lalu menggabungkan hasilnya melalui **Soft-Voting Ensemble** untuk memprediksi label severity: **Fatal**, **Mayor**, **Minor**, atau **Kosmetik**.

Pada tahap integrasi, antarmuka web "Form Tambah Log Temuan" terhubung ke API backend Python. Saat penguji mengetik deskripsi bug di kolom **Log Temuan**, sistem secara otomatis menganalisis teks menggunakan model yang sudah terlatih dan mengisi dropdown **Kategori Temuan** secara real-time — tanpa memerlukan proses re-training.

## System Flow

![System Flow](images/systemflow.png)

## Struktur Proyek

```
simata-bug-classifier/
├── data/
│   ├── raw/                     # CSV mentah (train_data.csv, test_data.csv)
│   ├── processed/               # Output preprocessing (*_cleaned.csv, *_embeddings.npy)
│   ├── output/                  # Hasil prediksi & evaluasi
│   │   ├── test_prediction.csv
│   │   ├── evaluation_results.csv
│   │   └── evaluation_summary.json
│   ├── dictionaries/            # kamus_slang.csv, id_freq_dict.txt
│   └── test/                    # Folder eksperimen
├── models/                      # Artifact model tersimpan
│   ├── indobert_severity.pt
│   ├── tfidf_vectorizer.pkl
│   ├── linearsvc_model.pkl
│   ├── xgboost_model.pkl
│   ├── label_encoder.pkl
│   └── ensemble_config.json
├── preprocessing/
│   ├── __init__.py
│   ├── pipeline.py              # Orchestrator preprocessing (2 tahap)
│   ├── text_cleaner.py          # Tahap 1: Regex cleaning (10 sub-step)
│   ├── slang_normalizer.py      # Normalisasi slang Bahasa Indonesia
│   └── feature_extractor.py     # Tahap 2: IndoBERT embedding
├── web/                         # Frontend web UI
│   ├── index.html               # Halaman form "Tambah Log Temuan"
│   ├── style.css                # Styling (severity colors, layout)
│   └── app.js                   # Logic (debounce, fetch API, update DOM)
├── notebooks/
│   └── 03_model_training.ipynb
├── main.py                      # CLI entry point (training pipeline)
├── trainer.py                   # Training & evaluasi ketiga model
├── config.py                    # Konfigurasi global (path, parameter)
├── api_server.py                # FastAPI server untuk prediksi real-time
├── evaluate_api.py              # Script evaluasi model terhadap test.csv
├── test.csv                     # Data test untuk evaluasi
└── requirements.txt
```

## Instalasi

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pastikan model artifacts tersedia

Semua file berikut harus ada di folder `models/`:

| File | Keterangan |
|------|------------|
| `indobert_severity.pt` | Weights fine-tuned IndoBERT classifier |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer (unigram + bigram) |
| `linearsvc_model.pkl` | Calibrated LinearSVC model |
| `xgboost_model.pkl` | XGBoost classifier |
| `label_encoder.pkl` | Label encoder (4 kelas) |
| `ensemble_config.json` | Bobot ensemble & konfigurasi |

> Jika model belum ada, jalankan training terlebih dahulu via `python main.py`.

## Cara Penggunaan

### A. Menjalankan Web UI (Rekomendasi Real-Time)

```bash
# Jalankan API server
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000

# Buka browser
# http://127.0.0.1:8000
```

**Alur penggunaan:**
1. Server memuat semua model artifacts saat startup (~10–15 detik)
2. Buka `http://127.0.0.1:8000` di browser
3. Ketik deskripsi bug di kolom **Log Temuan**
4. Setelah 500ms berhenti mengetik, sistem otomatis menganalisis teks
5. Dropdown **Kategori Temuan** terisi otomatis berdasarkan prediksi AI
6. User tetap dapat mengubah kategori secara manual melalui dropdown

**API Endpoints:**

| Method | Endpoint | Keterangan |
|--------|----------|------------|
| `GET` | `/` | Halaman web form |
| `GET` | `/api/health` | Status server & model |
| `POST` | `/api/predict` | Prediksi severity dari teks |

**Contoh request prediksi:**

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Error 500 saat klik tombol export"}'
```

**Contoh response:**

```json
{
  "prediction": "Mayor",
  "confidence": 0.7913,
  "probabilities": {
    "Fatal": 0.0683,
    "Kosmetik": 0.0344,
    "Mayor": 0.7913,
    "Minor": 0.1060
  },
  "cleaned_text": "error 500 saat klik tombol export"
}
```

### B. Training Model (CLI)

```bash
python main.py
```

Muncul menu:

```
[1] Upload file CSV baru (Train + Test)
[2] Gunakan data yang sudah ada (langsung training)
[0] Keluar
```

- Pilih `[1]` untuk upload `train_data.csv` dan `test_data.csv` baru via file dialog, lalu preprocessing otomatis dijalankan sebelum training.
- Pilih `[2]` jika data di `data/processed/` sudah ada, langsung lanjut ke training.

### C. Evaluasi Model

```bash
# Pastikan API server sedang berjalan, lalu:
python evaluate_api.py
```

Script ini mengirim setiap baris dari `test.csv` ke API `/api/predict`, membandingkan hasil prediksi dengan label sebenarnya, dan menghasilkan:
- **Classification report** (precision, recall, F1 per kelas)
- **Confusion matrix**
- **Detail kesalahan klasifikasi**
- Output disimpan di `data/output/evaluation_results.csv` dan `data/output/evaluation_summary.json`

### D. Penggunaan sebagai Library

```python
from preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline()

# Proses satu teks
result = pipeline.preprocess_text("1)tombolSimpan error500 tdk berfungsi")
print(result)
# → "tombol simpan error 500 tidak berfungsi"

# Proses CSV langsung
result_df = pipeline.preprocess_csv(
    input_path="data/raw/train_data.csv",
    output_path="data/processed/train_cleaned.csv"
)
```

## Format CSV

| Kolom | Keterangan |
|-------|------------|
| `Log Temuan` | Teks bug report mentah dari tester (wajib) |
| `Kategori` | Label severity (wajib di train, opsional di test) |

## Konfigurasi

Edit `config.py` untuk mengatur:

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `INDOBERT_MODEL_NAME` | `indobenchmark/indobert-base-p1` | Model HuggingFace yang digunakan |
| `INDOBERT_MAX_LENGTH` | `128` | Panjang maksimum token |
| `INDOBERT_BATCH_SIZE` | `32` | Ukuran batch saat embedding |
| `INPUT_COLUMN` | `Log Temuan` | Nama kolom teks input |
| `LABEL_COLUMN` | `Kategori` | Nama kolom label |
| `PIPELINE_STAGES` | keduanya `True` | Toggle aktif/nonaktif tiap tahap |

## Teknologi

| Komponen | Library | Peran |
|----------|---------|-------|
| Text Cleaning | `re` (built-in) | 10 sub-tahap regex: HTML preservation, CamelCase splitting, URL removal, dll |
| Slang Normalization | `pandas`, `dict` | Hash map O(1) lookup dari `kamus_slang.csv` |
| Feature Extraction | `transformers`, `torch` | IndoBERT encoder — embedding 768-dim per teks |
| BERT Classifier | `transformers`, `torch` | Fine-tuned classification head, 4 epoch, AdamW + LinearWarmup |
| TF-IDF | `scikit-learn` | Unigram + bigram vectorizer, 10.000 fitur, sublinear TF |
| LinearSVC | `scikit-learn` | SVM linear dengan `CalibratedClassifierCV` (cv=5) untuk probabilitas |
| XGBoost | `xgboost` | Gradient Boosted Trees, 200 estimator |
| Ensemble | `numpy` | Soft-voting: 50% IndoBERT + 25% LinearSVC + 25% XGBoost |
| Web Server | `fastapi`, `uvicorn` | API backend untuk prediksi real-time |
| Frontend | HTML, CSS, JS | Form Tambah Log Temuan dengan auto-select dropdown |

## Output

### Prediksi

Hasil prediksi disimpan di `data/output/test_prediction.csv`:

| Kolom | Isi |
|-------|-----|
| `Log Temuan` | Teks asli dari tester |
| `Kategori` | Label prediksi: Fatal / Mayor / Minor / Kosmetik |
| `cleaned_text` | Teks setelah preprocessing |

### Evaluasi

Hasil evaluasi disimpan di `data/output/evaluation_results.csv` dan `data/output/evaluation_summary.json`.

Model artifacts tersimpan di `models/` dan dapat digunakan kembali tanpa training ulang.
