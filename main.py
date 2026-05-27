"""
Exam Prep Bot - FastAPI Backend Application
Main application entry point with all API routes
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import asyncio
import json
from typing import List, Optional
from datetime import datetime
import logging

from src.models import (
    IntentClassificationResult, AnswerWithSources
)
from src.api_models import (
    QueryRequest, QueryResponse, DocumentUploadResponse
)
from src.pipeline import ExamPrepBot
from config.settings import settings

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Global bot instance
bot_instance: Optional[ExamPrepBot] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI"""
    # Startup
    global bot_instance
    logger.info("Initializing Exam Prep Bot...")
    bot_instance = ExamPrepBot()
    logger.info("Bot initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Exam Prep Bot...")
    if bot_instance:
        bot_instance.reset()
    logger.info("Bot shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title="📚 Exam Prep Bot API",
    description="Intelligent study assistant powered by Claude + RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== HEALTH CHECK ====================
@app.get("/", tags=["UI"])
async def root():
    """Root endpoint - serve frontend UI"""
    return FileResponse("frontend/index.html")

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "bot_initialized": bot_instance is not None
    }

# ==================== DOCUMENT MANAGEMENT ====================
@app.post("/api/documents/upload", tags=["Documents"], response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF or DOCX document.
    
    - **file**: PDF or DOCX file to upload
    
    Returns:
    - success: Whether upload succeeded
    - file_name: Name of uploaded file
    - total_chunks: Number of chunks created
    - processing_time_seconds: Time taken to process
    - message: Status message
    """
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    
    try:
        # Validate file type
        if file.filename.endswith('.pdf'):
            file_type = 'pdf'
        elif file.filename.endswith('.docx'):
            file_type = 'docx'
        else:
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files allowed")
        
        # Save file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            
            # Process document
            result = bot_instance.upload_document(tmp.name, file_type)
            
            return DocumentUploadResponse(**result)
    
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/documents/stats", tags=["Documents"])
async def get_document_stats():
    """Get statistics about uploaded documents"""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    
    stats = bot_instance.get_stats()
    return {
        "vector_store": stats['vector_store'],
        "chat_history_length": stats['chat_history_length'],
        "model": stats['model'],
        "embedding_model": stats['embedding_model'],
        "timestamp": datetime.now().isoformat()
    }

# ==================== QUERY & ANSWERING ====================
@app.post("/api/query", tags=["Query"], response_model=QueryResponse)
async def answer_question(request: QueryRequest):
    """
    Ask a question about uploaded documents.
    
    Supports 7 question types:
    - **definition**: "What is X?"
    - **explain**: "Detail me about X"
    - **compare**: "Compare X vs Y"
    - **process**: "What are the steps?"
    - **example**: "Give an example"
    - **diagram**: "Explain this diagram"
    - **vague**: "Answer this"
    
    Returns:
    - answer: The generated answer
    - intent: Classified query intent
    - sources: List of source citations
    - confidence: Overall confidence (0-1)
    - hallucination_risk: low/medium/high
    - response_time: Time taken to generate answer
    """
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    
    try:
        # Process query
        answer_with_sources = bot_instance.answer_question(request.query)
        
        # Convert to response format
        return QueryResponse(
            success=True,
            query=request.query,
            answer=answer_with_sources.answer,
            intent=answer_with_sources.query_intent.value,
            intent_confidence=answer_with_sources.intent_confidence,
            sources=[
                {
                    "page": s.page_number,
                    "section": s.section_title,
                    "quote": s.quoted_text,
                    "confidence": s.confidence,
                    "relevance": s.relevance_score
                }
                for s in answer_with_sources.sources
            ],
            confidence=answer_with_sources.overall_confidence,
            hallucination_risk=answer_with_sources.hallucination_risk,
            response_time_seconds=answer_with_sources.response_time_seconds,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/api/query/intent/{query}", tags=["Query"])
async def classify_intent(query: str):
    """
    Classify query intent without generating answer.
    
    Returns:
    - primary_intent: Main intent classification
    - confidence: Classification confidence (0-1)
    - alternative_intents: Other possible intents
    """
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    
    try:
        intent_result = bot_instance.intent_classifier.classify(query)
        
        return {
            "query": query,
            "primary_intent": intent_result.primary_intent.value,
            "confidence": intent_result.confidence,
            "alternative_intents": {
                k.value: v for k, v in (intent_result.alternative_intents or {}).items()
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# ==================== WEBSOCKET FOR STREAMING ====================
@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket endpoint for real-time query streaming.
    
    Message format (JSON):
    {
        "query": "Your question here",
        "stream": true
    }
    
    Response format (streaming):
    {
        "type": "intent" | "chunk" | "complete" | "error",
        "data": {...}
    }
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            query = message.get("query", "")
            
            if not query:
                await websocket.send_json({"type": "error", "message": "Query is required"})
                continue
            
            # Send intent classification
            try:
                intent_result = bot_instance.intent_classifier.classify(query)
                await websocket.send_json({
                    "type": "intent",
                    "intent": intent_result.primary_intent.value,
                    "confidence": intent_result.confidence
                })
            except Exception as e:
                logger.error(f"Intent classification error: {e}")
            
            # Process query
            try:
                answer_with_sources = bot_instance.answer_question(query)
                
                # Send chunks of answer
                chunk_size = 50
                for i in range(0, len(answer_with_sources.answer), chunk_size):
                    chunk = answer_with_sources.answer[i:i+chunk_size]
                    await websocket.send_json({
                        "type": "chunk",
                        "text": chunk
                    })
                
                # Send complete response
                await websocket.send_json({
                    "type": "complete",
                    "answer": answer_with_sources.answer,
                    "sources": [
                        {
                            "page": s.page_number,
                            "section": s.section_title,
                            "quote": s.quoted_text
                        }
                        for s in answer_with_sources.sources
                    ],
                    "confidence": answer_with_sources.overall_confidence,
                    "hallucination_risk": answer_with_sources.hallucination_risk,
                    "response_time": answer_with_sources.response_time_seconds
                })
            
            except Exception as e:
                logger.error(f"Query processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Query failed: {str(e)}"
                })
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# ==================== CHAT HISTORY ====================
@app.get("/api/chat/history", tags=["Chat"])
async def get_chat_history():
    """Get chat history"""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    
    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "intent": msg.intent_type.value if msg.intent_type else None
            }
            for msg in bot_instance.chat_history
        ],
        "total": len(bot_instance.chat_history)
    }

@app.delete("/api/chat/history", tags=["Chat"])
async def clear_chat_history():
    """Clear chat history"""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    
    bot_instance.chat_history = []
    
    return {"status": "success", "message": "Chat history cleared"}

# ==================== SYSTEM MANAGEMENT ====================
@app.post("/api/system/reset", tags=["System"])
async def reset_system():
    """Reset bot and clear all data"""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    
    try:
        bot_instance.reset()
        return {
            "status": "success",
            "message": "Bot reset successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.get("/api/system/config", tags=["System"])
async def get_config():
    """Get system configuration"""
    return {
        "model": settings.model_name,
        "embedding_model": settings.embedding_model,
        "max_chunk_size": settings.max_chunk_size,
        "retrieval_top_k": settings.retrieval_top_k,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "debug_mode": settings.debug_mode
    }

# ==================== BATCH OPERATIONS ====================
@app.post("/api/batch/query", tags=["Batch"])
async def batch_query(queries: List[str]):
    """
    Process multiple queries in batch.
    
    Returns list of answers for each query.
    """
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    
    results = []
    
    for query in queries:
        try:
            answer = bot_instance.answer_question(query)
            results.append({
                "query": query,
                "success": True,
                "answer": answer.answer,
                "confidence": answer.overall_confidence
            })
        except Exception as e:
            results.append({
                "query": query,
                "success": False,
                "error": str(e)
            })
    
    return {
        "total": len(queries),
        "successful": sum(1 for r in results if r.get("success", False)),
        "failed": sum(1 for r in results if not r.get("success", True)),
        "results": results
    }

# ==================== HEALTH & MONITORING ====================
@app.get("/api/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get system metrics and statistics"""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    
    stats = bot_instance.get_stats()
    
    return {
        "uptime": datetime.now().isoformat(),
        "vector_store": stats['vector_store'],
        "chat_history_length": stats['chat_history_length'],
        "model_info": {
            "model": stats['model'],
            "embedding_model": stats['embedding_model']
        }
    }

# Serve static files (for frontend)
# app.mount("/static", StaticFiles(directory="frontend/build"), name="static")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug_mode,
        log_level=settings.log_level.lower()
    )
