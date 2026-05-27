"""Query processing API endpoints"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from loguru import logger
import time
from typing import Optional

from app.models.schemas import QueryRequest, QueryResponse, QueryType
from app.services.pipeline import create_bot
from app.core.config import settings

router = APIRouter()

# Initialize bot
bot = create_bot()


@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask a question about uploaded documents
    
    - **query**: The question to ask
    - **document_id**: Optional specific document to search (if None, searches all)
    - **top_k**: Optional number of results (overrides default)
    """
    logger.info(f"Query received: {request.query}")
    
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    if len(request.query) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query too long (max 1000 characters)"
        )
    
    try:
        start_time = time.time()
        
        # Process query
        answer_with_sources = bot.answer_question(request.query)
        
        response_time = time.time() - start_time
        
        # Build response
        return QueryResponse(
            success=True,
            query=request.query,
            answer=answer_with_sources.answer,
            query_intent=answer_with_sources.query_intent,
            intent_confidence=answer_with_sources.intent_confidence,
            sources=answer_with_sources.sources,
            overall_confidence=answer_with_sources.overall_confidence,
            hallucination_risk=answer_with_sources.hallucination_risk,
            response_time_seconds=response_time,
            format_type=answer_with_sources.format_type
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/batch")
async def batch_queries(queries: list[str]):
    """
    Process multiple queries at once
    
    - **queries**: List of questions to ask
    """
    logger.info(f"Batch query received with {len(queries)} questions")
    
    if len(queries) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 50 queries per batch"
        )
    
    try:
        results = []
        
        for query in queries:
            try:
                request = QueryRequest(query=query)
                result = await ask_question(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query in batch: {e}")
                results.append({
                    "success": False,
                    "query": query,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "total_queries": len(queries),
            "successful": sum(1 for r in results if r.get("success", False)),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/history")
async def get_query_history(limit: int = 10, offset: int = 0):
    """
    Get query history
    
    - **limit**: Number of results to return
    - **offset**: Number of results to skip
    """
    logger.info(f"Getting query history (limit={limit}, offset={offset})")
    
    if limit > 100:
        limit = 100
    
    try:
        # In production, query database
        history = []
        
        return {
            "success": True,
            "total": 0,
            "limit": limit,
            "offset": offset,
            "history": history
        }
    
    except Exception as e:
        logger.error(f"Error getting query history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/intent-classify")
async def classify_intent(query: str):
    """
    Classify query intent without generating answer
    
    - **query**: The question to classify
    """
    logger.info(f"Intent classification request: {query}")
    
    try:
        from app.services.intent_classifier import IntentClassifier
        classifier = IntentClassifier()
        result = classifier.classify(query)
        
        return {
            "success": True,
            "query": query,
            "primary_intent": result.primary_intent,
            "confidence": result.confidence,
            "alternative_intents": result.alternative_intents
        }
    
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/search")
async def search_documents(query: str, top_k: Optional[int] = None):
    """
    Search documents without generating answer
    
    - **query**: Search query
    - **top_k**: Number of results to return
    """
    logger.info(f"Search request: {query}")
    
    try:
        from app.services.retriever import HybridRetriever
        from app.services.intent_classifier import IntentClassifier
        
        classifier = IntentClassifier()
        intent_result = classifier.classify(query)
        
        retriever = HybridRetriever()
        search_result = retriever.search(query, intent_result.primary_intent, top_k)
        
        return {
            "success": True,
            "query": query,
            "intent": search_result['intent'].value,
            "total_results": len(search_result['chunks']),
            "in_scope": search_result['in_scope'],
            "relevance_score": search_result['relevance_score'],
            "chunks": [
                {
                    "content": chunk.content[:200],  # First 200 chars
                    "page": chunk.metadata.page_number,
                    "relevance": chunk.relevance_score
                }
                for chunk in search_result['chunks']
            ]
        }
    
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
