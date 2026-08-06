"""Query processing API endpoints."""

import asyncio
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from app.core.config import redact_query_for_log
from app.core.dependencies import get_bot, get_intent_classifier, get_vector_store_manager
from app.core.rate_limit import limit_expensive
from app.core.security import require_api_key, require_api_key_ws
from app.models.schemas import BatchQueryRequest, QueryRequest, QueryResponse, SourceCitationOut

router = APIRouter()

# Shown to clients instead of the real exception — raw exception text can
# leak internal paths, DB connection details, or provider-side error
# payloads. The real detail always goes to the logs via logger.error/exception
# right above each use of this constant.
GENERIC_ERROR_DETAIL = "An internal error occurred. Please try again."


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
        session_id=request.session_id,
    )


async def _answer_query(payload: QueryRequest) -> QueryResponse:
    logger.info(
        f"[API /ask] query={redact_query_for_log(payload.query)}  document_id={payload.document_id}  "
        f"session_id={payload.session_id}  device_id={payload.device_id}"
    )
    bot = get_bot()
    try:
        answer = await run_in_threadpool(
            bot.answer_question,
            payload.query,
            session_id=payload.session_id,
            device_id=payload.device_id,
            document_ids=payload.document_ids,
        )
        response = _build_response(payload, answer)
        logger.info(
            f"[API /ask] OK: intent={response.query_intent}  confidence={response.overall_confidence:.3f}  "
            f"risk={response.hallucination_risk}  sources={len(response.sources)}  time={response.response_time_seconds:.3f}s"
        )
        return response
    except Exception as e:
        logger.error(
            f"[API /ask] FAILED: query={redact_query_for_log(payload.query)}  error={type(e).__name__}: {e}"
        )
        logger.exception("[API /ask] Full traceback:")
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_DETAIL)


@router.post("/ask", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
@limit_expensive()
async def ask_question(request: Request, payload: QueryRequest):
    return await _answer_query(payload)


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
@limit_expensive()
async def query_compat(request: Request, payload: QueryRequest):
    """Backward-compatible endpoint matching the original /api/query."""
    return await _answer_query(payload)


@router.get("/intent/{query}", dependencies=[Depends(require_api_key)])
async def classify_intent(query: str):
    logger.info(f"[API /intent] query={redact_query_for_log(query)}")
    classifier = get_intent_classifier()
    try:
        result = await run_in_threadpool(classifier.classify, query)
        logger.info(
            f"[API /intent] OK: intent={result.primary_intent.value}  confidence={result.confidence:.3f}"
        )
        return {
            "query": query,
            "primary_intent": result.primary_intent.value,
            "confidence": result.confidence,
            "alternative_intents": {k.value: v for k, v in (result.alternative_intents or {}).items()},
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(
            f"[API /intent] FAILED: query={redact_query_for_log(query)}  error={type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_DETAIL)


@router.post("/batch", dependencies=[Depends(require_api_key)])
@limit_expensive()
async def batch_queries(request: Request, payload: BatchQueryRequest):
    bot = get_bot()
    results: list[dict] = []
    for q in payload.queries:
        try:
            a = await run_in_threadpool(bot.answer_question, q)
            results.append(
                {"query": q, "success": True, "answer": a.answer, "confidence": a.overall_confidence}
            )
        except Exception as e:
            logger.error(
                f"[API /batch] item FAILED: query={redact_query_for_log(q)}  error={type(e).__name__}: {e}"
            )
            results.append({"query": q, "success": False, "error": GENERIC_ERROR_DETAIL})
    return {
        "total": len(payload.queries),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "results": results,
    }


@router.post("/search", dependencies=[Depends(require_api_key)])
async def search_documents(query: str, top_k: Optional[int] = None):
    classifier = get_intent_classifier()
    intent_result = await run_in_threadpool(classifier.classify, query)

    from app.services.retriever import HybridRetriever

    retriever = HybridRetriever(get_vector_store_manager())
    result = await run_in_threadpool(retriever.search, query, intent_result.primary_intent, top_k)
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
# Browsers cannot set custom headers on a WebSocket handshake, so auth here
# uses a ?api_key= query parameter instead of the X-API-Key header used by
# the REST endpoints above (see app.core.security.require_api_key_ws).


@router.websocket("/ws")
async def websocket_query(websocket: WebSocket):
    await websocket.accept()
    if not await require_api_key_ws(websocket):
        return

    bot = get_bot()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            query = message.get("query", "")
            session_id = message.get("session_id")
            device_id = message.get("device_id")
            document_ids = message.get("document_ids")
            domain_preset = message.get("domain_preset")
            top_k = message.get("top_k")
            temperature = message.get("temperature")
            if not query or not query.strip():
                await websocket.send_json({"type": "error", "message": "Query is required"})
                continue
            if len(query) > 1000:
                await websocket.send_json(
                    {"type": "error", "message": "Query must be 1000 characters or fewer"}
                )
                continue
            # Unlike the REST QueryRequest schema, nothing validates this JSON
            # payload before it reaches the pipeline — a malformed document_ids
            # (wrong type, non-string entries) would otherwise surface as an
            # opaque generic error deep inside run_in_threadpool.
            if document_ids is not None and (
                not isinstance(document_ids, list) or not all(isinstance(d, str) for d in document_ids)
            ):
                await websocket.send_json(
                    {"type": "error", "message": "document_ids must be a list of strings"}
                )
                continue

            try:
                intent_result = await run_in_threadpool(bot.intent_classifier.classify, query)
                await websocket.send_json(
                    {
                        "type": "intent",
                        "intent": intent_result.primary_intent.value,
                        "confidence": intent_result.confidence,
                    }
                )
            except Exception:
                pass

            try:
                # bot.answer_question runs in a worker thread (see
                # run_in_threadpool below); on_stage fires from that thread,
                # so the event is bridged onto the event loop via
                # call_soon_threadsafe into a queue and sent by a concurrent
                # drain task, fully flushed before "complete" is sent.
                #
                # Only one on_stage event ever fires now — "answer_ready",
                # carrying the final, already-reflected answer text (see
                # OrchestratorAgent.run). Drafting and reflection both happen
                # silently server-side first; nothing about that internal
                # pipeline is ever exposed to the client, which only ever
                # sees a generic "thinking" state (rendered client-side, no
                # server message needed) followed by this one streamed answer.
                loop = asyncio.get_running_loop()
                stage_queue: asyncio.Queue = asyncio.Queue()

                def on_stage(stage: str, payload: dict) -> None:
                    loop.call_soon_threadsafe(stage_queue.put_nowait, (stage, payload))

                # Paced so delivery reads as continuous natural typing rather
                # than the whole answer popping in at once — there is no
                # per-provider token stream to relay (see llm_interface.py),
                # so this pacing is what makes an already-fully-generated
                # string feel like it's streaming.
                async def _send_chunks(text: str, chunk_size: int = 4, delay_seconds: float = 0.01) -> None:
                    for i in range(0, len(text), chunk_size):
                        await websocket.send_json({"type": "chunk", "text": text[i : i + chunk_size]})
                        await asyncio.sleep(delay_seconds)

                async def _drain_stage_queue() -> None:
                    while True:
                        item = await stage_queue.get()
                        if item is None:
                            return
                        stage, payload = item
                        if stage == "answer_ready":
                            await _send_chunks(payload.get("answer", ""))

                drain_task = asyncio.create_task(_drain_stage_queue())
                try:
                    answer = await run_in_threadpool(
                        bot.answer_question,
                        query,
                        session_id=session_id,
                        device_id=device_id,
                        document_ids=document_ids,
                        on_stage=on_stage,
                    )
                finally:
                    await stage_queue.put(None)
                    await drain_task

                await websocket.send_json(
                    {
                        "type": "complete",
                        "answer": answer.answer,
                        "query_intent": answer.query_intent.value,
                        "format_type": answer.format_type,
                        "sources": [
                            {"page": s.page_number, "section": s.section_title, "quote": s.quoted_text}
                            for s in answer.sources
                        ],
                        "confidence": answer.overall_confidence,
                        "hallucination_risk": answer.hallucination_risk,
                        "response_time": answer.response_time_seconds,
                        "session_id": session_id,
                    }
                )
            except Exception as e:
                logger.error(f"[API /ws] turn FAILED: error={type(e).__name__}: {e}")
                logger.exception("[API /ws] Full traceback:")
                await websocket.send_json({"type": "error", "message": GENERIC_ERROR_DETAIL})
    except Exception:
        pass
    finally:
        await websocket.close()
