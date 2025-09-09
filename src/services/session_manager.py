import uuid
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from src.config import Config

@dataclass
class ChatMessage:
    """Representasi pesan dalam chat"""
    role: str  # "user" atau "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class ChatSession:
    """Representasi session chat dengan konteks percakapan"""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str):
        """Menambah pesan ke session"""
        message = ChatMessage(role=role, content=content)
        self.messages.append(message)
        self.last_activity = time.time()

        # Batasi jumlah pesan dalam konteks
        if len(self.messages) > Config.MAX_CONTEXT_LENGTH * 2:  # x2 karena ada user dan assistant
            # Hapus pesan lama, tapi pertahankan beberapa pesan terakhir
            self.messages = self.messages[-Config.MAX_CONTEXT_LENGTH:]

    def get_context_messages(self) -> List[Dict]:
        """Mendapatkan pesan-pesan untuk konteks LLM"""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def is_expired(self) -> bool:
        """Cek apakah session sudah expired"""
        return time.time() - self.last_activity > Config.SESSION_TIMEOUT

class SessionManager:
    """Manager untuk mengelola multiple chat sessions"""

    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}

    def create_session(self) -> str:
        """Membuat session baru dan return session_id"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ChatSession(session_id=session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Mendapatkan session berdasarkan ID"""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # Hapus session jika expired
        if session.is_expired():
            del self.sessions[session_id]
            return None

        return session

    def add_message_to_session(self, session_id: str, role: str, content: str) -> bool:
        """Menambah pesan ke session tertentu"""
        session = self.get_session(session_id)
        if not session:
            return False

        session.add_message(role, content)
        return True

    def get_session_context(self, session_id: str) -> List[Dict]:
        """Mendapatkan konteks percakapan dari session"""
        session = self.get_session(session_id)
        if not session:
            return []

        return session.get_context_messages()

    def cleanup_expired_sessions(self):
        """Membersihkan session-session yang expired"""
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.is_expired()
        ]

        for session_id in expired_sessions:
            del self.sessions[session_id]

        return len(expired_sessions)

    def get_active_sessions_count(self) -> int:
        """Mendapatkan jumlah session aktif"""
        self.cleanup_expired_sessions()
        return len(self.sessions)
