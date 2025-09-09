from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
from src.services.chatbot_service import ChatbotService
from src.config import Config

# Pydantic models untuk request/response
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    retrieved_documents: int
    confidence_scores: list
    sources: list

class SessionCreateResponse(BaseModel):
    session_id: str
    message: str

class ErrorResponse(BaseModel):
    error: str

# Initialize FastAPI app
app = FastAPI(
    title="Chatbot Unismuh API",
    description="API untuk sistem chatbot Universitas Muhammadiyah dengan arsitektur BERT + RAG + LLM",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot service instance
chatbot_service = ChatbotService()

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot service saat startup"""
    print("Starting chatbot service initialization...")
    try:
        # Initialize dalam background task agar tidak blocking
        chatbot_service.initialize()
        print("Chatbot service ready!")
    except Exception as e:
        print(f"Failed to initialize chatbot service: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup saat shutdown"""
    print("Shutting down chatbot service...")
    # Cleanup expired sessions
    chatbot_service.session_manager.cleanup_expired_sessions()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Chatbot Unismuh API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/sessions/create", response_model=SessionCreateResponse)
async def create_session():
    """Membuat session chat baru"""
    try:
        session_id = chatbot_service.create_session()
        return SessionCreateResponse(
            session_id=session_id,
            message="Session berhasil dibuat"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membuat session: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Mendapatkan informasi session"""
    try:
        session_info = chatbot_service.get_session_info(session_id)
        if "error" in session_info:
            raise HTTPException(status_code=404, detail=session_info["error"])
        return session_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint utama untuk chat"""
    try:
        if not chatbot_service.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Chatbot service belum ready. Tunggu sebentar dan coba lagi."
            )

        result = chatbot_service.chat(request.session_id, request.message)

        if "error" in result:
            if "Session tidak valid" in result["error"]:
                raise HTTPException(status_code=404, detail=result["error"])
            else:
                raise HTTPException(status_code=500, detail=result["error"])

        return ChatResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/system/stats")
async def get_system_stats():
    """Mendapatkan statistik sistem"""
    try:
        stats = chatbot_service.get_system_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/system/rebuild-index")
async def rebuild_index(background_tasks: BackgroundTasks):
    """Rebuild vector index (background task)"""
    def rebuild_task():
        try:
            chatbot_service.initialize(force_rebuild_index=True)
        except Exception as e:
            print(f"Error rebuilding index: {e}")

    background_tasks.add_task(rebuild_task)
    return {"message": "Index rebuild started in background"}

@app.post("/system/cleanup-sessions")
async def cleanup_sessions():
    """Cleanup expired sessions"""
    try:
        cleaned_count = chatbot_service.session_manager.cleanup_expired_sessions()
        return {
            "message": f"Cleaned up {cleaned_count} expired sessions",
            "cleaned_sessions": cleaned_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning sessions: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True
    )
