"""Query processing API endpoints."""

from fastapi import APIRouter, HTTPException, WebSocket, status
from loguru import logger
import json
import time
from typing import Optional
from datetime import datetime

from app.models.schemas import QueryRequest, QueryResponse, SourceCitationOut
from app.core.dependencies import get_bot, get_intent_classifier

router = APIRouter()


def _build_response(request: QueryRequest, answer_with_sources) -> QueryResponse:
    return QueryResponse(
        success=True,
        query=request.query,
        answer=answer_with_sources.answer,
        query_intent=answer_with_sources.query_intent.value,
        intent_confidence=answer_with_sources.intent_confidence,
        sources=[
            SourceCitationOut(
                page_number=s.page_number,
                section_title=s.section_title,
                quoted_text=s.quoted_text,
                confidence=s.confidence,
                relevance_score=s.relevance_score,
            )
            for s in answer_with_sources.sources
        ],
        overall_confidence=answer_with_sources.overall_confidence,
        hallucination_risk=answer_with_sources.hallucination_risk,
        response_time_seconds=answer_with_sources.response_time_seconds,
        format_type=answer_with_sources.format_type,
        timestamp=datetime.now().isoformat(),
    )


@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    bot = get_bot()
    try:
        answer = bot.answer_question(request.query)
        return _build_response(request, answer)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_compat(request: QueryRequest):
    """Backward-compatible endpoint matching the original /api/query."""
    return await ask_question(request)


@router.get("/intent/{query}")
async def classify_intent(query: str):
    classifier = get_intent_classifier()
    try:
        result = classifier.classify(query)
        return {
            "query": query,
            "primary_intent": result.primary_intent.value,
            "confidence": result.confidence,
            "alternative_intents": {k.value: v for k, v in (result.alternative_intents or {}).items()},
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def batch_queries(queries: list[str]):
    if len(queries) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 queries per batch")
    bot = get_bot()
    results = []
    for q in queries:
        try:
            a = bot.answer_question(q)
            results.append({"query": q, "success": True, "answer": a.answer, "confidence": a.overall_confidence})
        except Exception as e:
            results.append({"query": q, "success": False, "error": str(e)})
    return {
        "total": len(queries),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "results": results,
    }


@router.get("/history")
async def get_chat_history():
    bot = get_bot()
    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "intent": msg.intent_type.value if msg.intent_type else None,
            }
            for msg in bot.chat_history
        ],
        "total": len(bot.chat_history),
    }


@router.delete("/history")
async def clear_chat_history():
    bot = get_bot()
    bot.chat_history.clear()
    return {"status": "success", "message": "Chat history cleared"}


@router.post("/search")
async def search_documents(query: str, top_k: Optional[int] = None):
    classifier = get_intent_classifier()
    intent_result = classifier.classify(query)

    from app.services.retriever import HybridRetriever
    from app.core.dependencies import get_vector_store_manager

    retriever = HybridRetriever(get_vector_store_manager())
    result = retriever.search(query, intent_result.primary_intent, top_k)
    return {
        "success": True,
        "query": query,
        "intent": result["intent"].value,
        "total_results": len(result["chunks"]),
        "in_scope": result["in_scope"],
        "relevance_score": result["relevance_score"],
        "chunks": [
            {"content": c.content[:200], "page": c.metadata.page_number, "relevance": c.relevance_score}
            for c in result["chunks"]
        ],
    }


# ── WebSocket streaming ──────────────────────────────────────────────

@router.websocket("/ws")
async def websocket_query(websocket: WebSocket):
    await websocket.accept()
    bot = get_bot()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            query = message.get("query", "")
            if not query:
                await websocket.send_json({"type": "error", "message": "Query is required"})
                continue

            try:
                intent_result = bot.intent_classifier.classify(query)
                await websocket.send_json({
                    "type": "intent",
                    "intent": intent_result.primary_intent.value,
                    "confidence": intent_result.confidence,
                })
            except Exception:
                pass

            try:
                answer = bot.answer_question(query)
                chunk_size = 50
                for i in range(0, len(answer.answer), chunk_size):
                    await websocket.send_json({"type": "chunk", "text": answer.answer[i : i + chunk_size]})
                await websocket.send_json({
                    "type": "complete",
                    "answer": answer.answer,
                    "sources": [
                        {"page": s.page_number, "section": s.section_title, "quote": s.quoted_text}
                        for s in answer.sources
                    ],
                    "confidence": answer.overall_confidence,
                    "hallucination_risk": answer.hallucination_risk,
                    "response_time": answer.response_time_seconds,
                })
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
    except Exception:
        pass
    finally:
        await websocket.close()
