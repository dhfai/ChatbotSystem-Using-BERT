# 🧠 Arsitektur Lengkap Sistem Chatbot (BERT + RAG + LLM)
## Tujuan / Goal
Membangun sistem chatbot yang bukan hanya sekedar menjawab pertanyaan (rules-based), tetapi juga mampu memahami konteks pertanyaan maksud dari user:
1. **BERT**: Memahami konteks pertanyaan secara mendalam (semantic understanding).
2. **RAG**: Mengambil informasi relevan dari database besar (retrieval).
3. **LLM**: Menyajikan jawaban yang natural, koheren
4. **Dominasi BERT**: Menjadikan BERT sebagai model utama untuk pemahaman konteks, sehingga hasil retrieval lebih akurat dan relevan.
5. **RAG sebagai pengelola data**: RAG hanya bertugas mengambil dokumen relevan tanpa mengubah gaya bahasa atau menambah informasi.
6. **LLM sebagai naturalization layer**: LLM hanya digunakan untuk menyusun jawaban yang lebih manusiawi, tanpa menambah informasi baru.
7. **Minimalkan Hallucination**: Dengan struktur ini, diharapkan LLM tidak "ngarang" jawaban karena hanya menyajikan informasi yang sudah diambil oleh BERT + RAG.


## 1. Gambaran Umum

Sistem ini dirancang agar **BERT menjadi pusat utama (dominant model)** untuk pemahaman semantik dan retrieval, sementara **RAG hanya bertugas sebagai pengelola data**, dan **LLM dipakai sebagai naturalization layer** (penyaji jawaban yang lebih manusiawi).

```
Input User → BERT Layer → RAG → LLM → Output
```

---

## 2. Detail Arsitektur

### 🔹 2.1 Input Representation (di dalam BERT)

Setiap teks input (query user) melalui tahap:

* **Token Embedding** → representasi subword/token (misal WordPiece).
* **Segment Embedding** → menandai apakah token bagian dari kalimat A atau B.
* **Positional Embedding** → menandai posisi kata dalam kalimat (learned, bukan sinusoidal).

> Kombinasi ketiganya dijumlahkan untuk menghasilkan *input embedding* ke BERT.

---

### 🔹 2.2 BERT Encoder Stack

* **Jumlah Layer**:

  * BERT-base → 12 encoder layers (hidden size 768, 12 attention heads).
  * BERT-large → 24 encoder layers (hidden size 1024, 16 attention heads).

* **Setiap Encoder Layer terdiri dari:**

  1. **Multi-Head Self Attention (MHSA)**

     * Menghitung hubungan antar token (Query, Key, Value).
     * Formula:

       ```
       Attention(Q,K,V) = softmax( (QK^T) / sqrt(d_k) ) V
       ```
     * Multi-head memungkinkan model memahami konteks dari berbagai perspektif.

  2. **Add & Norm**

     * Residual connection + layer normalization untuk stabilisasi.

  3. **Feed-Forward Network (FFN)**

     * Dua fully connected layers + aktivasi (GELU).

  4. **Add & Norm (lagi)**

     * Supaya informasi dari input tetap terjaga.

* **Output Layer:**

  * Menghasilkan contextualized embedding untuk setiap token.
  * **\[CLS] token embedding** → dipakai sebagai representasi query.

---

### 🔹 2.3 Retrieval (RAG)

* **Langkah:**

  1. Query embedding (\[CLS] dari BERT) dicocokkan dengan dokumen embedding (hasil precompute).
  2. Metode similarity: **cosine similarity** atau **dot product**.
  3. Ambil **Top-K** dokumen paling relevan.

* **Fungsi:**

  * Menjadi *librarian* → hanya mengelola pengambilan data, tanpa mengubah bahasa/jawaban.

---

### 🔹 2.4 Naturalization Layer (LLM)

* **Input ke LLM:**

  * Query user.
  * Context hasil retrieval (Top-K dokumen).

* **Instruksi ke LLM:**

  * “Jawab hanya berdasarkan context ini.”
  * Jangan menambahkan informasi di luar retrieval.

* **Output:**

  * Jawaban yang lebih natural, koheren, mudah dipahami.
  * Tetap grounded pada hasil retrieval, sehingga tidak ngarang.

---

## 3. Alur Keseluruhan

```
User Input
   ↓
BERT Layer
   - Token + Segment + Positional Embedding
   - Transformer Encoder Layers
   - Output: [CLS] vector → Query Embedding
   ↓
RAG
   - Cari Top-K dokumen berdasarkan query embedding
   - Return context dokumen
   ↓
LLM
   - Ambil Query + Context
   - Susun jawaban natural (grounded)
   ↓
Output ke User
```

---

## 4. Kegunaan & Keunggulan

1. **Dominasi BERT**

   * Menjadi tulang punggung retrieval.
   * Bisa difine-tune dengan domain data spesifik (misal pendidikan, kesehatan, hukum).

2. **RAG sebagai pengelola data**

   * Tidak memengaruhi gaya bahasa.
   * Hanya fokus pada pemilihan dokumen relevan.

3. **LLM sebagai naturalizer**

   * Membuat jawaban enak dibaca tanpa mengubah fakta.
   * Meminimalkan *hallucination* karena tetap grounded ke hasil BERT + RAG.

4. **Modular & scalable**

   * Bisa diganti backend (FAISS, Qdrant, ChromaDB) untuk vector DB.
   * Bisa pakai LLM lokal (LLaMA, Mistral) atau API (GPT, Gemini).

---