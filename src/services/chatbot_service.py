from typing import Dict, List, Optional
from src.models.bert_embedder import BERTEmbedder
from src.models.rag_retriever import RAGRetriever
from src.models.llm_naturalizer import LLMNaturalizer
from src.services.session_manager import SessionManager
from src.data_processing.data_processor import DataProcessor
from src.config import Config
import os

class ChatbotService:
    """
    Service utama yang mengintegrasikan BERT + RAG + LLM
    Mengimplementasikan arsitektur sesuai docs.md
    """

    def __init__(self):
        self.bert_embedder = BERTEmbedder()
        self.rag_retriever = RAGRetriever()
        self.llm_naturalizer = LLMNaturalizer()
        self.session_manager = SessionManager()
        self.data_processor = DataProcessor()
        self.is_initialized = False

    def initialize(self, force_rebuild_index: bool = False):
        """
        Inisialisasi sistem chatbot
        """
        print("Initializing Chatbot Service...")

        # Load atau build vector index
        if force_rebuild_index or not self._index_exists():
            print("Building new vector index...")
            self._build_vector_index()
        else:
            print("Loading existing vector index...")
            self.rag_retriever.load_index()

        self.is_initialized = True
        print("Chatbot Service initialized successfully!")

    def _index_exists(self) -> bool:
        """
        Cek apakah vector index sudah ada
        """
        return (os.path.exists(Config.VECTOR_DB_INDEX_PATH) and
                os.path.exists(Config.VECTOR_DB_DOCUMENTS_PATH))

    def _build_vector_index(self):
        """
        Membangun vector index dari data
        """
        # Process data dari CSV dan Excel
        print("Processing data files...")
        self.data_processor.process_biaya_kuliah_data()
        self.data_processor.process_fakultas_data()

        # Tambahkan dokumen umum tentang universitas
        self._add_general_university_info()

        # Dapatkan semua dokumen
        documents, metadata = self.data_processor.get_all_documents()

        if not documents:
            raise RuntimeError("Tidak ada dokumen untuk di-index")

        # Buat vector index
        print("Creating vector index...")
        self.rag_retriever.create_index(documents, metadata)

        # Save index
        self.rag_retriever.save_index()

        print(f"Vector index created with {len(documents)} documents")

    def _add_general_university_info(self):
        """
        TIDAK menambahkan data apapun yang tidak ada di CSV/Excel asli
        Hanya menggunakan data dari file yang disediakan user
        """
        # HAPUS semua data tambahan yang tidak ada di CSV/Excel
        # Sistem hanya boleh menggunakan data dari:
        # 1. data_biaya_kuliah.csv
        # 2. data_fakultas.xlsx
        pass  # Tidak menambahkan data apapun

    def chat(self, session_id: str, user_message: str) -> Dict:
        """
        Proses chat utama mengikuti alur BERT → RAG → LLM
        Sesuai dengan arsitektur yang dirancang di docs.md
        """
        if not self.is_initialized:
            return {
                "error": "Chatbot belum diinisialisasi. Panggil initialize() terlebih dahulu."
            }

        try:
            # Validasi session
            session = self.session_manager.get_session(session_id)
            if not session:
                return {
                    "error": "Session tidak valid atau sudah expired. Buat session baru."
                }

            # === STEP 1: BERT LAYER (Dominant Model) ===
            print(f"STEP 1 - BERT Layer: Processing semantic understanding...")

            # Get conversation context untuk BERT contextual understanding
            conversation_context = self.session_manager.get_session_context(session_id)
            conversation_messages = [msg["content"] for msg in conversation_context[-3:]]

            # BERT sebagai dominant model: Encode query dengan contextual understanding
            query_embedding = self.bert_embedder.encode_query(user_message, conversation_messages)

            # BERT juga melakukan intent analysis (semantic understanding)
            intent_analysis = self.bert_embedder.encode_conversational_intent(user_message)

            print(f"BERT processed: Query embedded, Intent: {intent_analysis}")

            # === STEP 2: RAG LAYER (Data Manager) ===
            print(f"STEP 2 - RAG Layer: Retrieving relevant documents...")

            # RAG hanya bertugas sebagai pengelola data (sesuai docs.md)
            # Menggunakan BERT embedding untuk retrieval yang akurat
            retrieved_docs = self.rag_retriever.retrieve(
                user_message,
                conversation_context=conversation_messages
            )

            print(f"RAG retrieved: {len(retrieved_docs)} documents")

            # === STEP 3: LLM LAYER (Naturalization Layer) ===
            print(f"STEP 3 - LLM Layer: Naturalizing response...")

            # LLM hanya sebagai naturalization layer (sesuai docs.md)
            # Tidak menambah informasi, hanya menyusun jawaban yang manusiawi
            response = self.llm_naturalizer.naturalize_response(
                query=user_message,
                retrieved_documents=retrieved_docs,
                conversation_context=conversation_context
            )

            print(f"LLM naturalized: Response generated")

            # === STEP 4: OUTPUT PREPARATION ===
            # Update session dengan pesan baru
            self.session_manager.add_message_to_session(session_id, "user", user_message)
            self.session_manager.add_message_to_session(session_id, "assistant", response)

            # Prepare response dengan metadata lengkap
            result = {
                "session_id": session_id,
                "response": response,
                "retrieved_documents": len(retrieved_docs),
                "confidence_scores": [doc["score"] for doc in retrieved_docs[:3]],
                "sources": [doc["metadata"]["source"] for doc in retrieved_docs[:3]],
                # Tambahan metadata sesuai arsitektur
                "bert_intent_analysis": intent_analysis,
                "processing_steps": {
                    "bert_processing": True,
                    "rag_retrieval": True,
                    "llm_naturalization": True
                },
                "architecture_compliance": "BERT_dominant_RAG_LLM"
            }

            return result

        except Exception as e:
            error_message = str(e) if str(e) else repr(e)
            print(f"Error in chat processing: {error_message}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return {
                "error": f"Terjadi kesalahan: {error_message}"
            }

    def create_session(self) -> str:
        """
        Membuat session chat baru
        """
        return self.session_manager.create_session()

    def get_session_info(self, session_id: str) -> Dict:
        """
        Mendapatkan informasi session
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return {"error": "Session tidak ditemukan"}

        return {
            "session_id": session_id,
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "message_count": len(session.messages)
        }

    def get_system_stats(self) -> Dict:
        """
        Mendapatkan statistik sistem
        """
        return {
            "active_sessions": self.session_manager.get_active_sessions_count(),
            "is_initialized": self.is_initialized,
            "index_exists": self._index_exists()
        }
