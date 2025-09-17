# Dokumentasi Sistem Pengujian ChatbotSystem-Using-BERT

## Overview

Sistem pengujian ini dirancang untuk mengevaluasi performa ChatbotSystem-Using-BERT dengan berbagai metrik akurasi dan visualisasi. Terdiri dari 3 jenis pengujian utama:

1. **Pengujian 2 Pertanyaan** - Evaluasi dengan metrik BLEU, BERT Score, dan ROUGE
2. **Pengujian Session Percakapan** - Evaluasi 10 pertanyaan dalam 1 session dengan grafik
3. **Pengujian BERT Model** - Evaluasi model BERT dengan metrik akurasi, F1, recall, precision

## Struktur Folder Testing

```
testing/
├── __init__.py                     # Init module testing
├── chatbot_evaluator.py            # Core evaluator class
├── test_two_questions.py           # Test 2 pertanyaan
├── test_session_conversation.py    # Test session percakapan
├── test_bert_model.py              # Test evaluasi BERT model
├── run_all_tests.py                # Runner untuk semua test
├── requirements_testing.txt        # Dependencies testing
└── README.md                       # Dokumentasi ini
```

## Instalasi Dependencies

Sebelum menjalankan testing, install dependencies yang diperlukan:

```bash
# Aktifkan virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies testing
pip install -r testing/requirements_testing.txt

# Download NLTK data (jika diperlukan)
python -c "import nltk; nltk.download('punkt')"
```

## Cara Penggunaan

### 1. Menjalankan Semua Pengujian

```bash
# Menjalankan semua test sekaligus
python testing/run_all_tests.py

# Atau dengan argument eksplisit
python testing/run_all_tests.py --test all
```

### 2. Menjalankan Pengujian Spesifik

```bash
# Test 2 pertanyaan saja
python testing/run_all_tests.py --test two_questions

# Test session percakapan saja
python testing/run_all_tests.py --test session_conversation

# Test BERT model saja
python testing/run_all_tests.py --test bert_model
```

### 3. Menjalankan Test Individual

```bash
# Jalankan test individual langsung
python testing/test_two_questions.py
python testing/test_session_conversation.py
python testing/test_bert_model.py
```

## Detail Pengujian

### 1. Pengujian 2 Pertanyaan (`test_two_questions.py`)

**Tujuan**: Mengevaluasi akurasi chatbot dengan 2 pertanyaan berbeda menggunakan metrik BLEU, BERT Score, dan ROUGE.

**Metrik yang Dihitung**:
- BLEU Score
- ROUGE-1, ROUGE-2, ROUGE-L (F1, Precision, Recall)
- BERT Score (F1, Precision, Recall)

**Output**:
- `evaluation_two_questions_results.json`: Hasil evaluasi dalam format JSON
- Ringkasan metrik di console

### 2. Pengujian Session Percakapan (`test_session_conversation.py`)

**Tujuan**: Mengevaluasi percakapan berkelanjutan 10 pertanyaan dalam 1 session dengan visualisasi grafik.

**Pertanyaan yang Diuji**: 10 pertanyaan sesuai spesifikasi Anda tentang kuliah di UNISMUH.

**Metrik yang Dihitung**:
- BLEU Score per pertanyaan
- ROUGE scores per pertanyaan
- BERT Score per pertanyaan
- Rata-rata keseluruhan

**Output**:
- `session_evaluation_results.json`: Hasil evaluasi lengkap
- `session_evaluation_chart.png`: Grafik visualisasi 4 panel:
  - Metrik akurasi per pertanyaan (BLEU, ROUGE-1, BERT F1)
  - Variasi ROUGE scores
  - Komponen BERT Score
  - Rata-rata metrik keseluruhan

### 3. Pengujian BERT Model (`test_bert_model.py`)

**Tujuan**: Mengevaluasi performa BERT model sebagai dominant model dalam sistem.

**Metodologi**:
- Generate test data dari dokumen yang ada
- Hitung similarity antara query dan dokumen
- Evaluasi dengan berbagai threshold
- Analisis performa model

**Metrik yang Dihitung**:
- Accuracy
- Precision (Binary, Macro, Weighted)
- Recall (Binary, Macro, Weighted)
- F1-Score (Binary, Macro, Weighted)
- Classification Report

**Output**:
- `bert_model_evaluation_results.json`: Hasil evaluasi BERT
- `bert_threshold_analysis_results.json`: Analisis threshold optimal
- `bert_model_evaluation_chart.png`: Visualisasi 4 panel:
  - Metrik evaluasi utama
  - Perbandingan averaging methods
  - Distribusi data dan prediksi
  - Analisis threshold optimal

## Interpretasi Hasil

### Metrik BLEU
- Range: 0.0 - 1.0
- Semakin tinggi semakin baik
- Mengukur overlap n-gram antara jawaban sistem dan referensi

### Metrik ROUGE
- Range: 0.0 - 1.0
- Semakin tinggi semakin baik
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence

### BERT Score
- Range: 0.0 - 1.0
- Semakin tinggi semakin baik
- Mengukur similarity semantic menggunakan contextual embeddings

### Metrik Klasifikasi (BERT Model)
- **Accuracy**: Proporsi prediksi yang benar
- **Precision**: True Positive / (True Positive + False Positive)
- **Recall**: True Positive / (True Positive + False Negative)
- **F1-Score**: Harmonic mean of Precision dan Recall

## Troubleshooting

### Error Common

1. **ModuleNotFoundError**: Install dependencies dengan `pip install -r testing/requirements_testing.txt`

2. **NLTK Data Missing**: Download dengan `python -c "import nltk; nltk.download('punkt')"`

3. **CUDA/GPU Issues**: Model akan automatically fallback ke CPU

4. **Memory Issues**: Reduce batch size atau gunakan documents yang lebih sedikit

### Performance Tips

1. **Untuk Testing Cepat**: Edit jumlah samples di `test_bert_model.py`

2. **GPU Acceleration**: Ensure CUDA tersedia untuk faster embedding computation

3. **Parallel Processing**: Bisa diimplementasikan untuk multiple queries

## Kustomisasi

### Menambah Pertanyaan Test

Edit file `test_two_questions.py` atau `test_session_conversation.py` dan modifikasi:

```python
questions_with_expected = [
    ("Pertanyaan baru", "Expected answer baru"),
    # tambah pertanyaan lainnya
]
```

### Mengubah Threshold BERT

Edit di `test_bert_model.py`:

```python
# Ubah default threshold
results = evaluator.evaluate_bert_model(similarity_threshold=0.6)

# Atau ubah range threshold untuk analisis
threshold_results = evaluator.evaluate_different_thresholds([0.1, 0.3, 0.5, 0.7, 0.9])
```

### Custom Visualisasi

Modifikasi fungsi `create_visualization()` di masing-masing test file untuk mengubah style grafik.

## Output Files

Setelah menjalankan testing, akan dihasilkan file-file berikut di folder `testing/`:

1. `evaluation_two_questions_results.json`
2. `session_evaluation_results.json`
3. `session_evaluation_chart.png`
4. `bert_model_evaluation_results.json`
5. `bert_threshold_analysis_results.json`
6. `bert_model_evaluation_chart.png`
7. `complete_testing_report.json` (dari run_all_tests)

## Support

Jika mengalami masalah atau butuh modifikasi, check:

1. Pastikan chatbot service terinisialisasi dengan benar
2. Verify data JSON fakultas tersedia di folder `data/`
3. Check dependencies dan Python version compatibility
4. Review error logs untuk debugging

---

**Created by**: ChatbotSystem Testing Suite
**Version**: 1.0.0
**Last Updated**: 2025-09-16
