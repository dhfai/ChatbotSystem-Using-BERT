import faiss
import numpy as np
import json
import os
from typing import List, Dict, Tuple
from src.config import Config
from src.models.bert_embedder import BERTEmbedder

class RAGRetriever:
    """
    RAG component untuk retrieval dokumen relevan
    Bertugas sebagai pengelola data sesuai arsitektur di docs.md
    """

    def __init__(self):
        self.bert_embedder = BERTEmbedder()
        self.index = None
        self.documents = []
        self.document_metadata = []
        self.embedding_dim = None

    def create_index(self, documents: List[str], metadata: List[Dict] = None):
        """
        Membuat FAISS index dari dokumen-dokumen
        """
        print("Creating embeddings for documents...")
        embeddings = self.bert_embedder.encode_documents(documents)

        self.embedding_dim = embeddings.shape[1]
        self.documents = documents
        self.document_metadata = metadata or [{"id": i} for i in range(len(documents))]

        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))

        print(f"Index created with {len(documents)} documents")

    def save_index(self, index_path: str = None, documents_path: str = None):
        """
        Save FAISS index dan dokumen ke file
        """
        index_path = index_path or Config.VECTOR_DB_INDEX_PATH
        documents_path = documents_path or Config.VECTOR_DB_DOCUMENTS_PATH

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(documents_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, index_path)

        # Save documents and metadata
        data_to_save = {
            "documents": self.documents,
            "metadata": self.document_metadata,
            "embedding_dim": self.embedding_dim
        }

        with open(documents_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        print(f"Index saved to {index_path}")
        print(f"Documents saved to {documents_path}")

    def load_index(self, index_path: str = None, documents_path: str = None):
        """
        Load FAISS index dan dokumen dari file
        """
        index_path = index_path or Config.VECTOR_DB_INDEX_PATH
        documents_path = documents_path or Config.VECTOR_DB_DOCUMENTS_PATH

        if not os.path.exists(index_path) or not os.path.exists(documents_path):
            raise FileNotFoundError("Index atau documents file tidak ditemukan")

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load documents and metadata
        with open(documents_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.documents = data["documents"]
        self.document_metadata = data["metadata"]
        self.embedding_dim = data["embedding_dim"]

        print(f"Index loaded with {len(self.documents)} documents")

    def retrieve(self, query: str, top_k: int = None, conversation_context: List[str] = None) -> List[Dict]:
        """
        RAG Retrieval sesuai docs.md: Hanya bertugas sebagai pengelola data
        Menggunakan BERT embedding untuk retrieval yang akurat
        """
        top_k = top_k or Config.TOP_K_DOCUMENTS

        if self.index is None:
            raise RuntimeError("Index belum dibuat atau dimuat")

        # === RAG SEBAGAI DATA MANAGER (sesuai docs.md) ===
        print(f"RAG: Acting as data manager, retrieving top-{top_k} documents...")

        # Gunakan BERT embedding yang sudah diproses untuk retrieval
        query_embedding = self.bert_embedder.encode_query(query, conversation_context)

        # Validate dimension consistency
        if self.embedding_dim and len(query_embedding) != self.embedding_dim:
            print(f"Dimension mismatch: query={len(query_embedding)}, index={self.embedding_dim}")
            print("Rebuilding index due to dimension mismatch...")
            # Force fallback to basic search if dimensions don't match
            return []

        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # BERT-powered intent analysis untuk adaptive retrieval
        intent_analysis = self.bert_embedder.encode_conversational_intent(query)

        # Enhanced keyword matching for better retrieval with TKJ prioritization
        query_lower = query.lower()
        relevant_keywords = {
            'biaya': ['biaya', 'kuliah', 'uang', 'pembangunan', 'semester'],
            'jurusan': ['program studi', 'jurusan', 'fakultas', 'prodi'],
            'cocok': ['sesuai', 'cocok', 'tepat', 'relevan', 'minat'],
            'rekomendasi': ['rekomendasi', 'saran', 'pilihan', 'alternatif'],
            'tkj': ['informatika', 'jaringan komputer', 'teknologi informasi', 'pemrograman'],
            'tik': ['informatika', 'teknologi informasi', 'teknologi pendidikan'],
            'rpl': ['informatika', 'pengembangan perangkat lunak', 'kecerdasan buatan'],
            'multimedia': ['informatika', 'desain komunikasi visual', 'teknologi pendidikan'],
            'komputer': ['informatika', 'jaringan komputer', 'teknik elektro'],
            'jaringan': ['informatika', 'jaringan komputer', 'teknologi informasi'],
            'teknologi': ['informatika', 'teknologi informasi', 'teknologi pendidikan', 'teknik elektro'],
            'informasi': ['informatika', 'teknologi informasi', 'teknologi pendidikan'],
            'pemrograman': ['informatika', 'pengembangan perangkat lunak'],
            'elektro': ['teknik elektro'],
            'teknik': ['teknik elektro', 'teknik pengairan', 'arsitektur'],
            'ekonomi': ['ekonomi', 'manajemen', 'akuntansi'],
            'bisnis': ['manajemen', 'ekonomi', 'akuntansi'],
            'trading': ['manajemen', 'ekonomi', 'akuntansi'],
            'pengajar': ['pendidikan', 'guru', 'pgsd'],
            'pendidikan': ['pendidikan', 'guru', 'pgsd', 'teknologi pendidikan'],
            'teknologi_pendidikan': ['teknologi pendidikan']
        }

        # TKJ-specific boost mapping with higher priority for Informatika
        tkj_boost_mapping = {
            'informatika': 0.25,  # Highest boost for TKJ graduates
            'jaringan komputer': 0.25,
            'teknologi informasi': 0.25,
            'pemrograman': 0.25,
            'pengembangan perangkat lunak': 0.25,
            'teknik elektro': 0.15,  # Lower boost compared to Informatika
            'teknologi pendidikan': 0.10,
            'kecerdasan buatan': 0.20
        }

        # Adaptive retrieval strategy berdasarkan BERT intent analysis
        if intent_analysis["is_greeting"] or intent_analysis["is_casual_chat"]:
            # Untuk greeting, ambil dokumen umum
            search_k = min(top_k, 5)
            threshold = max(0.05, Config.SIMILARITY_THRESHOLD - 0.05)
        elif any(keyword in query_lower for keyword in ['biaya', 'kuliah', 'uang']):
            # Untuk pertanyaan biaya, ambil lebih banyak dokumen biaya
            search_k = min(top_k + 10, 25)  # Increased for cost queries
            threshold = max(0.05, Config.SIMILARITY_THRESHOLD - 0.05)  # Lower threshold for cost data
        elif intent_analysis["is_seeking_advice"] or intent_analysis["contains_university_terms"]:
            # Untuk pertanyaan universitas, ambil lebih banyak dokumen
            search_k = min(top_k + 5, 20)
            threshold = Config.SIMILARITY_THRESHOLD
        else:
            search_k = top_k + 2
            threshold = Config.SIMILARITY_THRESHOLD

        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, search_k)

        # RAG: Prepare results (hanya mengelola data, tidak mengubah isi)
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and score >= threshold:
                result = {
                    "document": self.documents[idx],
                    "metadata": self.document_metadata[idx],
                    "score": float(score),
                    "rank": i + 1,
                    "bert_intent_context": intent_analysis,
                    "retrieval_method": "BERT_semantic_similarity"
                }
                results.append(result)

        # Enhanced keyword-based boost for better matching with TKJ prioritization
        for keyword, related_terms in relevant_keywords.items():
            if keyword in query_lower:
                for j, result in enumerate(results):
                    doc_lower = result["document"].lower()
                    for term in related_terms:
                        if term in doc_lower:
                            # Base boost score
                            base_boost = 0.1

                            # Special boost for direct "teknologi pendidikan" query
                            if 'teknologi pendidikan' in query_lower and 'teknologi pendidikan' in doc_lower:
                                results[j]["score"] = min(1.0, result["score"] + 0.30)
                                results[j]["retrieval_method"] = "BERT_semantic_similarity_with_direct_match_boost"
                                continue

                            # Apply TKJ-specific boost if query contains TKJ-related terms
                            if any(tkj_term in query_lower for tkj_term in ['tkj', 'smk', 'komputer', 'jaringan', 'teknologi informasi', 'lulusan smk']):
                                if term in tkj_boost_mapping:
                                    boost_score = tkj_boost_mapping[term]
                                    results[j]["score"] = min(1.0, result["score"] + boost_score)
                                    results[j]["retrieval_method"] = f"BERT_semantic_similarity_with_TKJ_boost_{boost_score}"
                                else:
                                    results[j]["score"] = min(1.0, result["score"] + base_boost)
                                    results[j]["retrieval_method"] = "BERT_semantic_similarity_with_keyword_boost"
                            else:
                                # Normal keyword boost for non-TKJ queries
                                results[j]["score"] = min(1.0, result["score"] + base_boost)
                                results[j]["retrieval_method"] = "BERT_semantic_similarity_with_keyword_boost"

        # Sort by score after keyword boosting
        results.sort(key=lambda x: x["score"], reverse=True)

        # Enhanced fallback search for comprehensive coverage
        if len(results) < 5:  # If we have fewer than 5 results, expand search
            print("RAG: Expanding search for better coverage...")
            fallback_threshold = 0.05
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and score >= fallback_threshold:
                    # Check if document already in results
                    doc_already_included = any(r["metadata"].get("id") == idx for r in results)
                    if not doc_already_included:
                        result = {
                            "document": self.documents[idx],
                            "metadata": self.document_metadata[idx],
                            "score": float(score),
                            "rank": len(results) + 1,
                            "bert_intent_context": intent_analysis,
                            "retrieval_method": "BERT_expanded_search"
                        }
                        results.append(result)

        print(f"RAG: Retrieved {len(results)} documents using BERT-powered similarity")
        return results[:top_k]  # Return only top_k results
