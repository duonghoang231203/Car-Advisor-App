from fastapi import APIRouter, Depends, HTTPException, status, Body
from app.models.chat import ChatRequest, ChatResponse, ChatSession, Message
from app.services.rag_service import rag_service
from app.services.speech_service import speech_service
from app.db.mongodb import mongodb
from app.core.security import oauth2_scheme
from app.api.auth import oauth2_scheme
from bson import ObjectId
from typing import List, Optional
from datetime import datetime

router = APIRouter()

@router.post("/send", response_model=ChatResponse)
async def send_message(
    chat_request: ChatRequest,
    token: str = Depends(oauth2_scheme)
) -> ChatResponse:
    """
    Send a message to the chatbot and get a response
    """
    try:
        # Extract user_id from token
        # In a real implementation, you would decode the JWT token
        # For simplicity, we'll assume the token is the user_id
        user_id = "user123"  # Replace with actual user_id from token
        
        # Get or create chat session
        session_id = chat_request.session_id
        session = None
        
        if session_id:
            # Get existing session
            session_data = await mongodb.db.chat_sessions.find_one({"_id": ObjectId(session_id)})
            if session_data:
                session = ChatSession(**session_data)
        
        if not session:
            # Create new session
            session = ChatSession(user_id=user_id)
            result = await mongodb.db.chat_sessions.insert_one(session.dict(by_alias=True))
            session.id = result.inserted_id
            session_id = str(result.inserted_id)
        
        # Process image if provided
        image_context = ""
        if chat_request.image:
            # In a real implementation, you would process the image
            # For now, we'll just acknowledge it
            image_context = "I see you've shared an image of a car. "
        
        # Add user message to session
        user_message = Message(role="user", content=chat_request.message)
        session.messages.append(user_message)
        
        # Process query with RAG
        rag_result = await rag_service.process_query(chat_request.message)
        
        # Add assistant message to session
        assistant_message = Message(role="assistant", content=rag_result["response"])
        session.messages.append(assistant_message)
        
        # Update session in database
        session.updated_at = datetime.utcnow()
        await mongodb.db.chat_sessions.update_one(
            {"_id": session.id},
            {"$set": {
                "messages": [msg.dict() for msg in session.messages],
                "updated_at": session.updated_at
            }}
        )
        
        # Prepare response
        response = ChatResponse(
            response=image_context + rag_result["response"],
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
    token: str = Depends(oauth2_scheme)
) -> ChatResponse:
    """
    Process speech input and get a chatbot response
    
    audio_content should be base64 encoded audio
    """
    try:
        # Detect language
        language = await speech_service.detect_language(audio_content)
        
        # Transcribe audio
        transcribed_text = await speech_service.transcribe_audio(audio_content, language_code=language)
        
        if not transcribed_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not transcribe audio. Please try again."
            )
        
        # Create chat request with transcribed text
        chat_request = ChatRequest(
            message=transcribed_text,
            session_id=session_id
        )
        
        # Process chat request
        return await send_message(chat_request, token)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing speech: {str(e)}"
        )

@router.get("/history", response_model=List[Message])
async def get_chat_history(
    session_id: str,
    token: str = Depends(oauth2_scheme)
) -> List[Message]:
    """
    Get chat history for a specific session
    """
    try:
        # Extract user_id from token
        # In a real implementation, you would decode the JWT token
        user_id = "user123"  # Replace with actual user_id from token
        
        # Get session
        session_data = await mongodb.db.chat_sessions.find_one({
            "_id": ObjectId(session_id),
            "user_id": user_id
        })
        
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        
        # Return messages
        return [Message(**msg) for msg in session_data.get("messages", [])]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting chat history: {str(e)}"
        )

@router.get("/sessions")
async def get_chat_sessions(
    token: str = Depends(oauth2_scheme)
) -> List[dict]:
    """
    Get all chat sessions for the current user
    """
    try:
        # Extract user_id from token
        # In a real implementation, you would decode the JWT token
        user_id = "user123"  # Replace with actual user_id from token
        
        # Get sessions
        cursor = mongodb.db.chat_sessions.find({"user_id": user_id})
        sessions = await cursor.to_list(length=100)
        
        # Format response
        result = []
        for session in sessions:
            # Get first and last message for preview
            messages = session.get("messages", [])
            first_message = messages[0]["content"] if messages else ""
            last_message = messages[-1]["content"] if messages else ""
            
            result.append({
                "session_id": str(session["_id"]),
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
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