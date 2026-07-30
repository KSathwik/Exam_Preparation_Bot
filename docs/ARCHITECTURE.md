# Architecture Overview

This document covers the multi-agent RAG pipeline, memory architecture, and retrieval flow added in
the Phase 1 rebuild. For setup/commands and file-level pointers, see `CLAUDE.md`. For what's
deliberately deferred, see `PHASE_2_ROADMAP.md`.

## System overview

```mermaid
flowchart TB
    Client[Browser / API client] -->|REST /api/ask, /api/query, /api/batch| API[FastAPI routers<br/>app/api/*]
    Client -->|WebSocket /api/ws| API
    API --> Deps[app.core.dependencies<br/>DI singletons]
    Deps --> Bot[ExamPrepBot<br/>app/services/pipeline.py]
    Bot --> Orchestrator[OrchestratorAgent]
    Orchestrator --> Retrieval[RetrievalAgent]
    Orchestrator --> Knowledge[KnowledgeAgent]
    Orchestrator --> Reflection[ReflectionAgent]
    Orchestrator --> Memory[MemoryAgent]
    Retrieval --> VSM[VectorStoreManager<br/>FAISS + BM25]
    Knowledge --> LLM[_BaseLLM<br/>Anthropic / OpenAI / Gemini]
    Reflection --> LLM
    Memory --> DB[(SQLAlchemy DB<br/>ChatSession / ChatMessageRecord /<br/>ConversationMemory)]
    Memory -.persist=True only.-> VSM
```

## Agent workflow (per query)

`OrchestratorAgent.run()` is deterministic Python, not an LLM router — with only one downstream
answer-producing agent this phase (Knowledge), there's nothing meaningful to route between yet.
`IntentClassifier` already parameterizes the pipeline (top_k, prompt template, format_type); that's a
different thing from agent selection. Real LLM-based routing becomes justified once Phase 2 agents
(Quiz/StudyPlanner/...) exist to route between — see the roadmap doc.

```mermaid
sequenceDiagram
    participant O as OrchestratorAgent
    participant IC as IntentClassifier
    participant R as RetrievalAgent
    participant K as KnowledgeAgent
    participant V as SpanExtractor/ConfidenceScorer
    participant Rf as ReflectionAgent
    participant M as MemoryAgent

    O->>IC: classify(query)
    IC-->>O: intent, confidence
    O->>R: search(query, intent)
    R-->>O: chunks, is_relevant
    alt not is_relevant
        O-->>O: return out_of_scope fallback
    else in scope
        O->>K: generate(query, chunks, intent)
        K-->>O: draft_answer, claims [{claim, chunks}]
        O->>V: score(claims, chunks)
        V-->>O: citations, confidence, hallucination_risk
        O->>Rf: reflect(query, chunks, draft, validator_summary, intent)
        Rf-->>O: revised_answer, materially_changed, should_block, issues
        alt should_block
            O-->>O: return reflection_blocked fallback
        else materially_changed
            O->>V: re-score(revised_claims, chunks)
            V-->>O: updated citations/confidence
        end
        O->>M: record_turn(query, final_answer, intent)
        O-->>O: return AnswerWithSources
    end
```

Stage names fired via the optional `on_stage(stage, payload)` callback (used by the WebSocket route):
`retrieving` → `drafting` → `draft_ready` → `reflecting` → `final_ready`.

## WebSocket streaming contract (`/api/ws`)

Additive only — the REST `/api/ask` response shape (`AnswerWithSources`/`QueryResponse`) is untouched.

| Message | When | Shape |
|---|---|---|
| `intent` | After intent classification | `{"type": "intent", "intent": str, "confidence": float}` |
| `chunk` | Draft and final answer text, chunked | `{"type": "chunk", "text": str, "stage": "draft" \| "final"}` |
| `status` | Once, before the reflection LLM call | `{"type": "status", "stage": "reflecting", "message": str}` |
| `complete` | End of turn | unchanged: `answer`, `query_intent`, `format_type`, `sources`, `confidence`, `hallucination_risk`, `response_time` |
| `error` | Any failure | `{"type": "error", "message": str}` |

This is an honest two-stage progressive reveal (draft, then possibly-revised final) — not true
token-level LLM streaming, which is out of scope this phase.

## Retrieval pipeline

```mermaid
flowchart LR
    Q[Query] --> Dense[Dense search<br/>FAISS over-fetch]
    Q --> BM25[BM25Okapi<br/>lazily rebuilt in-memory]
    Dense --> Blend["alpha * dense_norm +<br/>(1-alpha) * bm25_norm"]
    BM25 --> Blend
    Blend --> CE{enable_cross_encoder_rerank?}
    CE -->|yes, opt-in| Rerank[CrossEncoder rerank]
    CE -->|no, default| Skip[skip]
    Rerank --> Intent[rerank_by_intent<br/>deterministic tie-break]
    Skip --> Intent
    Intent --> Scope{best_score >= relevance_threshold?}
    Scope -->|yes| Chunks[document chunks]
    Scope -->|no| MemSearch[semantic memory search<br/>content_types=memory]
    MemSearch --> MemScope{best_score >= memory_relevance_threshold?}
    MemScope -->|yes| MemChunks[memory chunks, in_scope=True]
    MemScope -->|no| OutOfScope[out_of_scope fallback]
```

Both FAISS and BM25 operate over the *same* shared index, discriminated by a `content_type` field
(`"document"` or `"memory"`) on each chunk's metadata — not two separate indices, to avoid duplicating
the persistence/locking/consistency-check machinery this project already hardened around FAISS.

## Chunking (parser.py)

```mermaid
flowchart TB
    Doc[Raw document] --> Blocks{File type}
    Blocks -->|DOCX| DocxBlocks[paragraph.style.name<br/>exact heading levels]
    Blocks -->|PDF| PdfBlocks[font-size heuristic per page<br/>falls back to one block/page]
    DocxBlocks --> Accum[Sentence-accumulate per block<br/>up to max_chunk_size words]
    PdfBlocks --> Accum
    Accum --> Frag{Trailing fragment<br/>< min_chunk_size?}
    Frag -->|prior chunk in same block| Merge[merge into it]
    Frag -->|no prior chunk, more blocks left| Carry[carry into next block]
    Frag -->|no prior chunk, last block| Force[force-emit as its own chunk]
    Frag -->|>= min_chunk_size| Emit[emit as a chunk]
```

A non-empty document never yields zero chunks — every trailing fragment lands somewhere.

## Conversation memory tiers

| Tier | Backing store | Access pattern | When invoked |
|---|---|---|---|
| Short-term | In-memory `chat_history` list | Verbatim recency window | Every turn (always active) |
| Episodic | `ChatMessageRecord` rows for one `session_id` | Direct lookup, chronological | Only on explicit reference (not yet wired to an endpoint) |
| Long-term/semantic | `ConversationMemory` rows, embedded (`content_type="memory"`) | `VectorStoreManager.search(content_types=["memory"])` | Only when document retrieval misses |

```mermaid
sequenceDiagram
    participant O as OrchestratorAgent
    participant M as MemoryAgent (persist=True)
    participant DB as SQLAlchemy DB
    participant LLM as _BaseLLM
    participant VSM as VectorStoreManager

    O->>M: record_turn(query, answer, intent)
    M->>DB: insert ChatMessageRecord (user + assistant)<br/>token_count = running total
    M->>DB: session.turn_count += 1
    alt turn_count % N == 0  OR  tokens since last summary >= threshold
        M->>DB: fetch messages since last_summarized_message_id
        M->>LLM: summarize_conversation(turns)
        LLM-->>M: summary_text
        M->>DB: insert ConversationMemory (embedded=False)
        M->>VSM: add_memory(summary_text, session_id, memory_id)
        VSM-->>M: ok
        M->>DB: memory.embedded = True
    end
```

`persist=True` is forward-compat plumbing, not the default — there is no per-user session concept
wired into `ExamPrepBot`/`get_bot()` yet (today's auth is a single shared `APP_API_KEY`). A memory row
is written *before* the embedding call, so a crash between the two never loses the summary — an
unembedded row (`embedded=False`) is simply invisible to semantic search and can be retried.
