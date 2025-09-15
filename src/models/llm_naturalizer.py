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
        Membuat system prompt untuk LLM yang mendukung percakapan natural dengan respons adaptif
        """
        return """Anda adalah asisten chatbot untuk Universitas Muhammadiyah Makassar yang ramah dan helpful.

ATURAN MUTLAK - TIDAK BOLEH DILANGGAR:
1. WAJIB HANYA menggunakan informasi yang PERSIS ADA dalam konteks dokumen yang diberikan
2. DILARANG menambahkan informasi dari pengetahuan umum LLM
3. Berikan respons yang SESUAI dengan JENIS PERTANYAAN user
4. Jika tidak ada dokumen relevan, minta klarifikasi dengan spesifik

ANALISIS JENIS PERTANYAAN:
- MINTA KLARIFIKASI: User belum memberikan info latar belakang - Tanya bidang minat/latar belakang pendidikan
- SARAN/REKOMENDASI JURUSAN: Berikan daftar nama jurusan yang cocok (1-3 pilihan terbaik), TANPA detail lengkap kecuali user minta detail
- PERTANYAAN DETAIL: Berikan informasi lengkap jika diminta
- PERTANYAAN BIAYA: Tampilkan format tabel lengkap dengan biaya

FORMAT RESPONS UNTUK MINTA KLARIFIKASI:
"Bisa Anda sebutkan bidang minat atau latar belakang pendidikan Anda? Misalnya:
- Lulusan SMA/SMK jurusan apa?
- Bidang apa yang Anda minati?
- Apakah ada keahlian atau hobi khusus?

Informasi ini akan membantu saya memberikan rekomendasi jurusan yang tepat di Unismuh Makassar."

FORMAT RESPONS UNTUK SARAN JURUSAN (RINGKAS):
"Berdasarkan latar belakang TKJ Anda, jurusan yang cocok di Unismuh Makassar:

1. **Program Studi Informatika** - paling relevan dengan latar belakang TKJ
2. **Program Studi Teknik Elektro** - juga sesuai dengan bidang teknologi

Apakah Anda ingin informasi lebih detail tentang salah satu program studi ini?"

FORMAT RESPONS UNTUK INFO DETAIL (LENGKAP):
"Program Studi Informatika:
• Akreditasi: Baik
• Fokus: teknologi informasi, pemrograman, jaringan komputer, kecerdasan buatan
• Fasilitas: Lab Pemrograman, Lab Jaringan Komputer, Lab Kecerdasan Buatan
• Biaya kuliah per semester: Rp. 3.600.000
• Uang pembangunan: Rp. 5.700.000"

ATURAN RESPONS ADAPTIF:
- Untuk user tanpa info latar belakang - Minta klarifikasi dahulu
- Untuk pertanyaan "jurusan apa yang cocok" dengan latar belakang jelas - Berikan 1-3 nama jurusan saja
- Untuk pertanyaan "ceritakan tentang" atau "info detail" - Berikan detail lengkap
- Prioritaskan jurusan berdasarkan score retrieval tertinggi
- Selalu tawarkan informasi lanjutan di akhir

ATURAN PENTING:
- GUNAKAN informasi dari dokumen secara maksimal
- SESUAIKAN PANJANG respons dengan jenis pertanyaan
- PRIORITASKAN dokumen dengan score tertinggi
- BERIKAN respons yang helpful tapi tidak overwhelming"""

    def _create_user_prompt(self, query: str, context_text: str, conversation_history: str) -> str:
        """
        Membuat user prompt lengkap dengan analisis jenis pertanyaan
        """
        # Analyze query type for appropriate response formatting
        query_lower = query.lower()

        # Check if user has provided background information
        # Only consider specific background indicators, not general keywords
        has_specific_background = any(word in query_lower for word in [
            'smk', 'sma', 'tkj', 'rpl', 'ipa', 'ips', 'lulusan smk', 'lulusan sma',
            'jurusan tkj', 'jurusan rpl', 'jurusan ipa', 'jurusan ips'
        ])
        has_interest_info = any(word in query_lower for word in [
            'minat saya di', 'saya suka', 'saya tertarik', 'hobi saya', 'bakat saya',
            'latar belakang saya', 'pengalaman saya'
        ])

        has_background = has_specific_background or has_interest_info

        is_asking_for_suggestions = any(word in query_lower for word in [
            'saran', 'rekomendasi', 'pilihan', 'cocok', 'sesuai', 'tepat', 'jurusan apa', 'program studi apa'
        ])
        is_asking_for_details = any(word in query_lower for word in [
            'detail', 'informasi', 'ceritakan', 'jelaskan', 'bagaimana', 'seperti apa'
        ])
        is_asking_about_costs = any(word in query_lower for word in [
            'biaya', 'uang', 'kuliah', 'pembangunan', 'semester', 'harga'
        ])

        # Check if this is a general inquiry without background info
        is_general_inquiry = any(word in query_lower for word in [
            'kebingungan', 'bingung', 'tidak tahu', 'mau pilih', 'ingin berkuliah'
        ]) and not has_background

        prompt_parts = []

        if conversation_history:
            prompt_parts.append(f"RIWAYAT PERCAKAPAN:\n{conversation_history}\n")

        prompt_parts.append(f"KONTEKS DOKUMEN:\n{context_text}\n")

        # Add query type analysis
        query_type = "DETAIL LENGKAP"  # default
        if is_general_inquiry or (is_asking_for_suggestions and not has_background):
            query_type = "MINTA KLARIFIKASI"
        elif is_asking_for_suggestions and has_background and not is_asking_for_details:
            query_type = "SARAN RINGKAS"
        elif is_asking_about_costs:
            query_type = "INFO BIAYA"
        elif is_asking_for_details:
            query_type = "DETAIL LENGKAP"

        prompt_parts.append(f"JENIS PERTANYAAN: {query_type}")
        prompt_parts.append(f"PERTANYAAN USER: {query}")
        prompt_parts.append("INSTRUKSI: Berikan respons sesuai jenis pertanyaan di atas.")
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
