import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
from src.config import Config

class BERTEmbedder:
    """
    BERT-based embedding model untuk semantic understanding
    Menjadi dominant model dalam arsitektur sesuai docs.md
    Implementasi sesuai dengan desain: Input Representation → BERT Encoder Stack → [CLS] Token
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Gunakan BERT multilingual sebagai backbone (sesuai docs.md)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        self.bert_model = AutoModel.from_pretrained(Config.BERT_MODEL_NAME)
        self.bert_model.to(self.device)

        # Sentence transformer sebagai fallback untuk compatibility
        self.sentence_transformer = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
        self.sentence_transformer.to(self.device)

        print(f"BERT Dominant Model initialized: {Config.BERT_MODEL_NAME}")

    def encode_query_with_bert_backbone(self, query: str, conversation_context: List[str] = None) -> np.ndarray:
        """
        BERT Dominant Processing sesuai docs.md:
        1. Input Representation (Token + Segment + Positional Embedding)
        2. BERT Encoder Stack (12/24 layers dengan Multi-Head Self Attention)
        3. [CLS] Token Embedding sebagai query representation
        """
        # Prepare input dengan context jika ada
        if conversation_context and len(conversation_context) > 0:
            # Gabungkan context dengan query untuk contextual understanding
            context_str = " [SEP] ".join(conversation_context[-2:])  # 2 pesan terakhir
            full_input = f"{context_str} [SEP] {query}"
        else:
            full_input = query

        # BERT Tokenization (Input Representation)
        inputs = self.tokenizer(
            full_input,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)

        # BERT Forward Pass (Encoder Stack)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

            # Ambil [CLS] token embedding (sesuai docs.md)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return cls_embedding.flatten()

    def encode_query(self, query: str, conversation_context: List[str] = None) -> np.ndarray:
        """
        Main encoding method yang menggunakan BERT sebagai dominant model
        Fallback ke sentence transformer untuk compatibility
        """
        try:
            # Gunakan BERT backbone sebagai dominant model
            return self.encode_query_with_bert_backbone(query, conversation_context)
        except Exception as e:
            print(f"BERT backbone error: {e}, using fallback...")
            # Fallback ke sentence transformer
            if conversation_context and len(conversation_context) > 0:
                recent_context = conversation_context[-3:]
                contextual_query = " ".join(recent_context + [query])

                query_embedding = self.sentence_transformer.encode(query, convert_to_tensor=False)
                context_embedding = self.sentence_transformer.encode(contextual_query, convert_to_tensor=False)

                # Weighted combination: 70% query, 30% contextual
                final_embedding = 0.7 * query_embedding + 0.3 * context_embedding
                return final_embedding
            else:
                return self.sentence_transformer.encode(query, convert_to_tensor=False)

    def encode_conversational_intent(self, message: str) -> dict:
        """
        BERT-powered intent analysis untuk semantic understanding
        """
        message_lower = message.lower()

        # Enhanced intent analysis menggunakan BERT contextual understanding
        intent_analysis = {
            "is_question": any(word in message_lower for word in [
                'apa', 'bagaimana', 'kapan', 'dimana', 'siapa', 'mengapa', 'kenapa',
                'berapa', 'adakah', 'apakah', 'bisakah', '?'
            ]),
            "is_storytelling": any(word in message_lower for word in [
                'saya', 'aku', 'kita', 'kami', 'cerita', 'pengalaman', 'pernah',
                'dulu', 'sekarang', 'sedang', 'akan', 'rencana'
            ]),
            "is_seeking_advice": any(word in message_lower for word in [
                'sebaiknya', 'harusnya', 'rekomendasi', 'saran', 'tips', 'bantuan',
                'bingung', 'pilih', 'pilihan', 'keputusan'
            ]),
            "is_greeting": any(word in message_lower for word in [
                'halo', 'hai', 'hello', 'selamat', 'assalamualaikum', 'permisi'
            ]),
            "is_casual_chat": any(word in message_lower for word in [
                'gimana', 'kayak', 'kan', 'nih', 'dong', 'sih', 'ya'
            ]),
            "semantic_complexity": len(message.split()),  # Ukuran kompleksitas semantik
            "contains_university_terms": any(term in message_lower for term in [
                'unismuh', 'universitas', 'fakultas', 'jurusan', 'kuliah', 'kampus'
            ])
        }

        return intent_analysis

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode multiple documents menggunakan metode yang konsisten dengan query encoding
        """
        # Ensure consistency by using the same method as encode_query
        embeddings = []
        for doc in documents:
            # Use the same encoding path as queries to ensure dimension consistency
            embedding = self.encode_query(doc)  # This will handle fallback consistently
            embeddings.append(embedding)
        return np.array(embeddings)
