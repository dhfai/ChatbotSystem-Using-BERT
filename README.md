# 🤖 Chatbot Universitas Muhammadiyah

Sistem chatbot cerdas menggunakan arsitektur **BERT + RAG + LLM** untuk menjawab pertanyaan tentang Universitas Muhammadiyah dengan pemahaman konteks yang mendalam.

## 🏗️ Arsitektur Sistem

Berdasarkan dokumentasi di `docs.md`, sistem ini mengimplementasikan:

1. **BERT** - Dominant model untuk semantic understanding
2. **RAG** - Retrieval-Augmented Generation untuk mengelola data
3. **LLM** - Naturalization layer untuk response yang manusiawi
4. **Session Management** - Menangani konteks percakapan

```
Input User → BERT Layer → RAG Retrieval → LLM Naturalization → Output
```

## 📁 Struktur Project

```
chatbot-unismuh/
├── src/
│   ├── models/
│   │   ├── bert_embedder.py      # BERT untuk semantic understanding
│   │   ├── rag_retriever.py      # RAG retrieval system
│   │   └── llm_naturalizer.py    # LLM naturalization layer
│   ├── services/
│   │   ├── chatbot_service.py    # Main chatbot service
│   │   └── session_manager.py    # Session management
│   ├── api/
│   │   └── main.py              # FastAPI application
│   ├── data_processing/
│   │   └── data_processor.py    # Data processing utilities
│   └── config.py                # Configuration settings
├── data/
│   ├── data_fakultas.csv        # Data fakultas
│   └── data_biaya_kuliah.csv    # Data biaya kuliah
├── run.py                       # Main runner script
├── initialize.py                # System initialization
└── requirements.txt             # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Environment (Opsional)

```bash
cp .env .env
# Edit .env dan tambahkan OPENAI_API_KEY jika ada
```

### 3. Initialize System

```bash
python run.py init
```

### 4. Run API Server

```bash
python run.py api
```

API akan berjalan di `http://localhost:8000`

Dokumentasi API: `http://localhost:8000/docs`

## 📖 Penggunaan API

### 1. Buat Session Baru

```bash
curl -X POST "http://localhost:8000/sessions/create"
```

Response:
```json
{
  "session_id": "uuid-session-id",
  "message": "Session berhasil dibuat"
}
```

### 2. Chat dengan Bot

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "message": "Apa saja fakultas yang ada di Universitas Muhammadiyah?"
  }'
```

Response:
```json
{
  "session_id": "your-session-id",
  "response": "Berdasarkan data yang tersedia, Universitas Muhammadiyah memiliki beberapa fakultas...",
  "retrieved_documents": 5,
  "confidence_scores": [0.85, 0.78, 0.72],
  "sources": ["fakultas", "fakultas", "university_info"]
}
```

### 3. Cek System Stats

```bash
curl "http://localhost:8000/system/stats"
```

## 🛠️ Command Line Usage

### Initialize System
```bash
python run.py init                    # Normal initialization
python run.py init --force-rebuild    # Force rebuild vector index
```

### Run API Server
```bash
python run.py api                     # Run with default settings
python run.py api --host 0.0.0.0 --port 8080  # Custom host/port
python run.py api --no-reload         # Disable auto-reload
```

### Test System
```bash
python run.py test                    # Run basic functionality test
```

## 🔧 Konfigurasi

Edit `src/config.py` untuk menyesuaikan:

- Model BERT yang digunakan
- Konfigurasi vector database
- Pengaturan LLM
- Timeout session
- Dan lainnya

## 📊 Data Sources

Sistem menggunakan data dari:

1. **data_fakultas.csv** - Informasi fakultas dan program studi
2. **data_biaya_kuliah.csv** - Informasi biaya kuliah
3. **Custom documents** - Informasi umum universitas

## 🎯 Fitur Utama

- ✅ **Semantic Understanding** dengan BERT
- ✅ **Context-Aware Retrieval** dengan RAG
- ✅ **Natural Language Generation** dengan LLM
- ✅ **Session Management** untuk konteks percakapan
- ✅ **RESTful API** dengan FastAPI
- ✅ **Modular Architecture** untuk easy maintenance
- ✅ **Fallback Response** jika LLM tidak tersedia

## 🔄 Workflow Internal

1. **User Input** → Diterima via API
2. **BERT Processing** → Query di-encode untuk semantic understanding
3. **RAG Retrieval** → Cari dokumen relevan dari vector database
4. **Context Building** → Gabungkan retrieved docs + conversation history
5. **LLM Naturalization** → Generate response yang natural
6. **Session Update** → Simpan ke session untuk konteks selanjutnya
7. **Response** → Kirim ke user

## 🚨 Troubleshooting

### Error: "Chatbot belum diinisialisasi"
```bash
python run.py init
```

### Error: "Session tidak valid"
Buat session baru:
```bash
curl -X POST "http://localhost:8000/sessions/create"
```

### Error: Dependencies missing
```bash
pip install -r requirements.txt
```

### Vector index error
```bash
python run.py init --force-rebuild
```

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/sessions/create` | Buat session baru |
| GET | `/sessions/{session_id}` | Info session |
| POST | `/chat` | Chat dengan bot |
| GET | `/system/stats` | System statistics |
| POST | `/system/rebuild-index` | Rebuild vector index |
| POST | `/system/cleanup-sessions` | Cleanup expired sessions |

## 🧪 Testing

```bash
# Test basic functionality
python run.py test

# Test specific component
python initialize.py
```

## 🔮 Future Enhancements

- [ ] Support multiple languages
- [ ] Web interface
- [ ] Analytics dashboard
- [ ] Custom model fine-tuning
- [ ] Database integration
- [ ] Batch processing

## 📞 Support

Untuk pertanyaan atau issues, silakan buat issue di repository ini.

---

**Made with ❤️ for Universitas Muhammadiyah**
