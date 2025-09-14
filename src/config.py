import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # BERT Model Configuration
    BERT_MODEL_NAME = "bert-base-multilingual-cased"
    SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Vector Database Configuration
    VECTOR_DB_INDEX_PATH = "data/vector_index.faiss"
    VECTOR_DB_DOCUMENTS_PATH = "data/documents.json"

    # LLM Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL = "gpt-3.5-turbo"

    # Session Configuration
    SESSION_TIMEOUT = 3600  # 1 hour in seconds
    MAX_CONTEXT_LENGTH = 10  # Maximum previous messages to keep in context

    # Data Paths
    DATA_FOLDER = "data"
    DATA_JSON_FILES = ["data_feb.json", "data_fkip.json", "data_ft.json"]

    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000

    # Retrieval Configuration
    TOP_K_DOCUMENTS = 15  # Increased from 8 to 15 for better coverage
    SIMILARITY_THRESHOLD = 0.1  # Lowered from 0.15 to 0.1 for even better retrieval
