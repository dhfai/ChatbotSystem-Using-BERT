from openai import OpenAI
from typing import List, Dict, Optional
from src.config import Config

class LLMNaturalizer:
    """
    LLM component sebagai naturalization layer
    Hanya untuk menyusun jawaban yang lebih manusiawi sesuai arsitektur di docs.md
    """

    def __init__(self):
        if Config.OPENAI_API_KEY:
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        else:
            self.client = None
            print("Warning: OPENAI_API_KEY tidak ditemukan. LLM tidak akan berfungsi.")

    def naturalize_response(self, query: str, retrieved_documents: List[Dict],
                          conversation_context: List[Dict] = None) -> str:
        """
        Menggunakan LLM untuk menyusun jawaban natural berdasarkan retrieved documents
        Tidak menambah informasi baru, hanya naturalisasi
        """
        if not self.client:
            return self._fallback_response(retrieved_documents)

        # Prepare context from retrieved documents
        context_text = self._prepare_context(retrieved_documents)

        # Prepare conversation history
        conversation_history = self._prepare_conversation_history(conversation_context)

        # Create system prompt
        system_prompt = self._create_system_prompt()

        # Create user prompt
        user_prompt = self._create_user_prompt(query, context_text, conversation_history)

        try:
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            error_message = str(e) if str(e) else repr(e)
            print(f"Error calling LLM: {error_message}")
            print(f"Falling back to basic response due to LLM error")
            return self._fallback_response(retrieved_documents)

    def _prepare_context(self, retrieved_documents: List[Dict]) -> str:
        """
        Menyiapkan konteks dari dokumen yang di-retrieve
        """
        if not retrieved_documents:
            return "Tidak ada informasi yang relevan ditemukan."

        context_parts = []
        for i, doc in enumerate(retrieved_documents, 1):
            context_parts.append(f"Dokumen {i} (skor: {doc['score']:.3f}):\n{doc['document']}")

        return "\n\n".join(context_parts)

    def _prepare_conversation_history(self, conversation_context: List[Dict] = None) -> str:
        """
        Menyiapkan riwayat percakapan untuk konteks
        """
        if not conversation_context:
            return ""

        history_parts = []
        for msg in conversation_context[-5:]:  # Ambil 5 pesan terakhir
            role = "User" if msg["role"] == "user" else "Assistant"
            history_parts.append(f"{role}: {msg['content']}")

        return "\n".join(history_parts) if history_parts else ""

    def _create_system_prompt(self) -> str:
        """
        Membuat system prompt untuk LLM yang mendukung percakapan natural
        """
        return """Anda adalah asisten chatbot untuk Universitas Muhammadiyah Makassar yang ramah dan helpful. 

ATURAN MUTLAK - TIDAK BOLEH DILANGGAR:
1. WAJIB HANYA menggunakan informasi yang PERSIS ADA dalam konteks dokumen yang diberikan
2. DILARANG menambahkan informasi dari pengetahuan umum LLM
3. Jika ada dokumen relevan, berikan informasi lengkap dari dokumen tersebut
4. Jika tidak ada dokumen relevan, minta klarifikasi dengan spesifik

CARA MENJAWAB YANG BENAR:
- Jika ada dokumen dengan informasi biaya/program studi, tampilkan dalam format yang jelas
- Berikan informasi konkret dan lengkap dari dokumen
- Jangan ragu memberikan data faktual yang ada dalam dokumen
- Format informasi dalam bentuk list yang mudah dibaca

UNTUK PERTANYAAN TENTANG PROGRAM STUDI:
- Jika dokumen mengandung informasi program studi, tampilkan dengan lengkap
- Include: Nama program, akreditasi, biaya kuliah per semester, uang pembangunan
- Jika banyak program, tampilkan yang paling relevan (5-10 program)
- Berikan penjelasan singkat tentang struktur biaya

UNTUK PERTANYAAN UMUM TANPA KONTEKS:
- Minta informasi spesifik: "Bisa Anda sebutkan bidang minat atau latar belakang pendidikan Anda?"
- Tawarkan eksplorasi: "Saya dapat membantu dengan informasi program studi yang spesifik"

FORMAT RESPONS YANG DIINGINKAN:
1. Jawaban langsung berdasarkan dokumen yang ada
2. Informasi terstruktur dan mudah dibaca
3. Data lengkap jika tersedia dalam dokumen
4. Penawaran bantuan lebih lanjut

CONTOH FORMAT UNTUK INFO BIAYA:
"Berikut beberapa program studi yang mungkin sesuai:

1. **S-1 Informatika**
   - Akreditasi: Baik
   - Biaya kuliah per semester: Rp. 3.600.000
   - Uang pembangunan: Rp. 5.700.000

2. **S-1 Manajemen**
   - Akreditasi: Unggul
   - Biaya kuliah per semester: Rp. 3.500.000
   - Uang pembangunan: Rp. 5.000.000

*Uang pembangunan dibayar sekali di semester pertama.
Apakah ada program studi tertentu yang ingin Anda ketahui lebih detail?"

ATURAN PENTING:
- GUNAKAN informasi dari dokumen secara maksimal
- TAMPILKAN data lengkap jika tersedia
- JANGAN terlalu konservatif jika data sudah jelas ada
- BERIKAN respons yang helpful dan informatif"""

    def _create_user_prompt(self, query: str, context_text: str, conversation_history: str) -> str:
        """
        Membuat user prompt lengkap
        """
        prompt_parts = []

        if conversation_history:
            prompt_parts.append(f"RIWAYAT PERCAKAPAN:\n{conversation_history}\n")

        prompt_parts.append(f"KONTEKS DOKUMEN:\n{context_text}\n")
        prompt_parts.append(f"PERTANYAAN USER: {query}\n")
        prompt_parts.append("JAWABAN:")

        return "\n".join(prompt_parts)

    def _fallback_response(self, retrieved_documents: List[Dict]) -> str:
        """
        Response fallback jika LLM tidak tersedia
        """
        if not retrieved_documents:
            return "Maaf, saya tidak menemukan informasi yang relevan untuk pertanyaan Anda."

        response_parts = ["Berdasarkan informasi yang saya temukan:\n"]

        for i, doc in enumerate(retrieved_documents[:3], 1):
            response_parts.append(f"{i}. {doc['document'][:200]}...")

        return "\n".join(response_parts)
