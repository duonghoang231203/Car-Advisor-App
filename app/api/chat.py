from fastapi import APIRouter, Depends, HTTPException, status, Body
from app.models.chat import ChatRequest, ChatResponse, ChatSession, Message
from app.services.rag_service import rag_service
from app.services.speech_service import speech_service
# Authentication removed
# from app.core.security import oauth2_scheme
# from app.api.auth import oauth2_scheme
from bson import ObjectId
from typing import List, Optional
from datetime import datetime

router = APIRouter()

# In-memory storage for chat sessions (for demo/testing only)
chat_sessions = {}

@router.post("/send", response_model=ChatResponse)
async def send_message(
    chat_request: ChatRequest,
    # token: str = Depends(oauth2_scheme)
) -> ChatResponse:
    """
    Send a message to the chatbot and get a response
    """
    try:
        # Extract user_id from token
        user_id = "user123"  # Replace with actual user_id from token
        session_id = chat_request.session_id
        session = None

        if session_id and session_id in chat_sessions:
            session = chat_sessions[session_id]
        if not session:
            session = ChatSession(user_id=user_id)
            session_id = str(id(session))
            session.id = session_id
            chat_sessions[session_id] = session

        # Add user message to session
        user_message = Message(role="user", content=chat_request.message)
        session.messages.append(user_message)

        # Process query with RAG, including conversation history
        # Only pass the last 5 messages to avoid context window limitations
        conversation_history = session.messages[-5:] if len(session.messages) > 0 else []
        rag_result = await rag_service.process_query(
            query=chat_request.message,
            conversation_history=conversation_history
        )

        # Add assistant message to session
        assistant_message = Message(role="assistant", content=rag_result["response"])
        session.messages.append(assistant_message)

        # Update session in memory
        session.updated_at = datetime.utcnow()
        chat_sessions[session_id] = session

        # Prepare response
        response = ChatResponse(
            response=rag_result["response"],
            suggestions=rag_result["suggestions"],
            explanation=rag_result["explanation"],
            session_id=str(session.id)
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )

@router.post("/speech", response_model=ChatResponse)
async def process_speech(
    audio_content: str = Body(..., embed=True),
    session_id: Optional[str] = Body(None, embed=True),
) -> ChatResponse:
    """
    Process speech input and get a chatbot response
    audio_content should be base64 encoded audio
    """
    try:
        language = await speech_service.detect_language(audio_content)
        transcribed_text = await speech_service.transcribe_audio(audio_content, language_code=language)
        if not transcribed_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not transcribe audio. Please try again."
            )
        chat_request = ChatRequest(
            message=transcribed_text,
            session_id=session_id
        )
        return await send_message(chat_request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing speech: {str(e)}"
        )

@router.get("/history", response_model=List[Message])
async def get_chat_history(
    session_id: str
) -> List[Message]:
    """
    Get chat history for a specific session
    """
    try:
        user_id = "user123"  # Replace with actual user_id from token
        session = chat_sessions.get(session_id)
        if not session or session.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        return session.messages
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting chat history: {str(e)}"
        )

@router.get("/sessions")
async def get_chat_sessions() -> List[dict]:
    """
    Get all chat sessions for the current user
    """
    try:
        user_id = "user123"  # Replace with actual user_id from token
        result = []
        for session in chat_sessions.values():
            if session.user_id != user_id:
                continue
            messages = session.messages
            first_message = messages[0].content if messages else ""
            last_message = messages[-1].content if messages else ""
            result.append({
                "session_id": str(session.id),
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "message_count": len(messages),
                "preview": first_message[:50] + "..." if len(first_message) > 50 else first_message,
                "last_message": last_message[:50] + "..." if len(last_message) > 50 else last_message
            })
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting chat sessions: {str(e)}"
        )
