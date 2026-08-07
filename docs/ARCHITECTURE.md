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
    Deps --> Bot[KnowledgeAssistantPipeline<br/>app/services/pipeline.py]
    Bot --> Orchestrator[OrchestratorAgent]
    Orchestrator --> IntentC[IntentClassifier<br/>what it's about]
    Orchestrator --> FormatC[FormatClassifier<br/>how it should look]
    Orchestrator --> ContextR[ContextRouter<br/>RAG vs CAG]
    Orchestrator --> Retrieval[RetrievalAgent]
    Orchestrator --> Knowledge[KnowledgeAgent]
    Orchestrator --> Reflection[ReflectionAgent]
    Orchestrator --> Formatter[response_formatter.format_response]
    Orchestrator --> Memory[MemoryAgent]
    ContextR -.decides strategy.-> Retrieval
    Retrieval --> VSM[VectorStoreManager<br/>FAISS + BM25]
    Knowledge --> LLM[_BaseLLM<br/>Anthropic / OpenAI / Gemini / Ollama]
    Reflection --> LLM
    Memory --> DB[(SQLAlchemy DB<br/>ChatSession / ChatMessageRecord /<br/>ConversationMemory)]
    Memory -.persist=True only.-> VSM
```

`IntentClassifier`, `FormatClassifier`, and `ContextRouter` are all deterministic, LLM-free decisions —
none of them are agent routing (see below).

## Agent workflow (per query)

`OrchestratorAgent.run()` is deterministic Python, not an LLM router — with only one downstream
answer-producing agent this phase (Knowledge), there's nothing meaningful to route between yet.
Three orthogonal, fully deterministic (no LLM call) decisions parameterize the pipeline instead:
`IntentClassifier` (what the question is about — retrieval `top_k`/reranking), `FormatClassifier` (how
the answer should look — drafting prompt template + final formatting), and `ContextRouter` (RAG vs CAG
— ranked retrieval vs whole-document context). None of them is agent selection. Real LLM-based routing
becomes justified once Phase 2 agents (Quiz/StudyPlanner/...) exist to route between — see the roadmap
doc.

```mermaid
sequenceDiagram
    participant O as OrchestratorAgent
    participant IC as IntentClassifier
    participant FC as FormatClassifier
    participant CR as ContextRouter
    participant R as RetrievalAgent
    participant K as KnowledgeAgent
    participant V as SpanExtractor/ConfidenceScorer
    participant Rf as ReflectionAgent
    participant Ft as response_formatter
    participant M as MemoryAgent

    O->>IC: classify(query)
    IC-->>O: intent, confidence
    O->>FC: classify(query, intent)
    FC-->>O: response_format
    alt scoped to conversation document(s)
        O->>R: get_full_context(document_ids)
        R-->>O: all chunks (unranked)
        O->>CR: decide(chunks)
        CR-->>O: RAG | CAG
    end
    alt CAG chosen
        Note over O,R: chunks already fetched above — reused as-is
    else RAG (default, or CAG scope too large/absent)
        O->>R: search(query, intent, document_ids)
        R-->>O: chunks, is_relevant
    end
    alt not is_relevant
        O-->>O: return out_of_scope fallback
    else in scope
        O->>K: generate(query, chunks, intent, response_format)
        K-->>O: draft_answer, claims [{claim, chunks}]
        O->>V: score(claims, chunks)
        V-->>O: citations, confidence, hallucination_risk
        alt citation_rate==0 AND confidence << block floor
            O-->>O: pre-reflection shortcut — return hard-block fallback, skip Rf entirely
        else
            O->>Rf: reflect(query, chunks, draft, validator_summary, intent, response_format)
            Rf-->>O: revised_answer, materially_changed, should_block, issues
            alt should_block
                O-->>O: return reflection_blocked fallback
            else materially_changed
                O->>V: re-score(revised_claims, chunks)
                V-->>O: updated citations/confidence
            end
            alt confidence < 0.3 OR citation_rate == 0
                O-->>O: return hard-block fallback (hallucination-risk gate)
            else
                O->>Ft: format_response(final_answer, response_format)
                Ft-->>O: cleaned final_answer
                O->>M: record_turn(query, final_answer, intent, format_type)
                O-->>O: return AnswerWithSources
            end
        end
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
token-level LLM streaming, which is out of scope this phase. `format_type`'s shape (a plain string) is
unchanged since intent-driven formatting shipped — only its *value set* grew from a handful of
intent-derived labels to ~20 `ResponseFormat` values (`key_points`, `summary`, `comparison`, `mcq`,
`flashcards`, ...), so no frontend/schema change was needed to add it.

## Hybrid RAG + CAG retrieval routing

`ContextRouter.decide()` (called once per turn, only when the query resolves to a real, non-empty
conversation document scope) picks between two retrieval strategies before either one runs — not a
fallback chain, a decision made from a single already-fetched chunk list:

```mermaid
flowchart TB
    Scope{Resolved document scope?} -->|none / global search| RAGPath[RAG path — see below]
    Scope -->|non-empty| Fetch[get_chunks_by_document_ids<br/>all chunks, unranked, page+chunk_index order]
    Fetch --> Budget{enable_cag AND<br/>estimated_tokens <= cag_token_budget?}
    Budget -->|yes| CAGPath[CAG: hand every fetched chunk<br/>straight to the draft call]
    Budget -->|no| RAGPath
```

Everything from Stage 3 (draft generation) onward is identical regardless of which branch produced the
`List[RetrievedChunk]` — CAG is a second way to *populate* Stage 2's output, not a parallel pipeline.
The real-world common case for this app is small documents (single-digit chunk counts per upload), so
CAG is the typical path in practice, not RAG.

## Retrieval pipeline (RAG path)

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

    O->>M: record_turn(query, answer, intent, format_type)
    M->>DB: insert ChatMessageRecord (user + assistant)<br/>token_count = running total<br/>format_type set on assistant row only
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
