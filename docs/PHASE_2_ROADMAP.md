# Phase 2 Roadmap

Phase 1 (this rebuild) delivered the multi-agent pipeline foundation: Orchestrator/Retrieval/
Knowledge/Reflection/Memory agents, hybrid (dense + BM25) retrieval with opt-in cross-encoder
reranking, structural chunking, chunk-indexed citations, and an opt-in conversation-memory
persistence layer — since extended with intent-driven response formatting (`format_classifier.py`/
`response_formats.py`, ~20 presentation formats detected from ordinary phrasing) and Hybrid RAG + CAG
retrieval routing (`context_router.py`, whole-document context instead of ranked chunks when the
scope is small enough). Everything below was scoped out of Phase 1 by explicit decision, not because
it's unimportant — this is a punch list for the next phase, not a wishlist.

## Deferred agents

> **Not to be confused with**: intent-driven response *formatting* (`response_formats.py`) already
> lets a student ask for MCQ-shaped or flashcard-shaped *output* today ("generate MCQs on this",
> "create flashcards for this chapter") — that's a presentation format on top of the existing Q&A
> pipeline, produced in one LLM call like any other answer. The agents below are a different, bigger
> thing: standalone generation *with grading/scheduling/spaced-repetition state*, not just shaping a
> single answer's Markdown.

- **Quiz Agent** — generate practice questions (MCQ/short-answer) from uploaded material, graded
  against the source excerpts the same way answers are cited today.
- **Flashcard Agent** — spaced-repetition card generation from document chunks or conversation
  memory summaries.
- **NotesGenerator Agent** — condensed study notes per document/topic, likely reusing
  `_BaseLLM.summarize_conversation`'s pattern but over document chunks instead of chat turns.
- **StudyPlanner Agent** — schedules review sessions across documents/topics; the first agent that
  would make an LLM-based *routing* layer in `OrchestratorAgent` actually justified (today there's
  only one downstream agent — Knowledge — so a router has nothing to decide between).
- **Citation-as-a-formal-agent** — `SpanExtractor`/`ConfidenceScorer` remain plain classes invoked
  directly by the orchestrator (no LLM call, so "agent" status wasn't warranted this phase); revisit
  if citation logic grows an LLM-based verification step.

## Multi-tenancy & JWT/OAuth2 Authentication Architecture

Phase 1 established forward-compatible schema and API key foundations:
- `ChatSession.user_id` and `ConversationMemory.user_id` are nullable columns, added specifically so real multi-tenant auth won't require DB schema migrations.
- Today's REST/WebSocket security uses `X-API-Key` headers and browser-scoped `device_id` identifiers in local storage for session isolation.

**Phase 2 Multi-Tenant Migration Blueprint**:
1. **User Identity & JWT/OAuth2 Provider**: Integrate Auth0 / Keycloak / NextAuth or FastAPI OAuth2 (`OAuth2PasswordBearer` + JWT tokens).
2. **Bearer Token Middleware**: Replace single shared `APP_API_KEY` requirement on protected routes with `Authorization: Bearer <jwt_token>` validation (`app/core/security.py`).
3. **User Isolation**: Automatically bind document uploads, vector store namespaces, and conversation history directly to the authenticated `user_id` extracted from JWT claims.

## Progress tracking

Depends on the Quiz/Flashcard agents existing first (there's nothing to track progress *on* yet
otherwise) — e.g. mastery-per-topic derived from quiz results, spaced-repetition scheduling state,
and a dashboard surfacing it. Revisit once those agents land.

## Retrieval/embedding follow-ups (reassess, don't build speculatively)

- **Provider-level prompt/context caching** — Hybrid RAG + CAG (`context_router.py`) already gives the
  drafting LLM a whole small document's context instead of ranked chunks; the next, designed-but-not-
  built step is reusing that *same* context across a conversation's repeat queries without
  re-processing it, via each provider's native caching (Anthropic `cache_control` content blocks,
  Gemini's explicit `client.caches.create`/`cached_content=`, OpenAI's automatic caching for prompts
  ≥1024 tokens). Deliberately not bundled with the CAG routing work itself — it's provider-specific,
  Gemini's version needs real new state (a cache-handle registry keyed by document-set + model, with
  TTL tracking), and the routing change alone already captures the main accuracy/latency win.
- **FAISS index type**: currently `IndexFlatL2` — query speed is a non-issue at this project's scale
  (low-single-digit ms even at tens of thousands of vectors). Reassess **only** once a single shared
  index exceeds roughly 200k vectors; at that point `IndexIVFFlat` (not `IndexHNSWFlat`) is the
  correct next step, since it's the only index type that improves both query speed *and*
  delete-by-id — `IndexHNSWFlat`'s graph structure doesn't help the actual pain point
  (`FAISSVectorStore.remove_by_document_id` rebuilding the whole index on every removal).
- **Multilingual embeddings**: if non-English study material turns out to matter,
  `intfloat/multilingual-e5-small` is the named fallback to `BAAI/bge-small-en-v1.5` — but it needs
  `"query: "`/`"passage: "` prefix handling added to `EmbeddingGenerator`, an actual code change
  (unlike the BGE swap, which was a pure config default change). Not built speculatively.
- **Cross-encoder reranking always-on**: currently opt-in and off by default
  (`settings.enable_cross_encoder_rerank`) because hybrid retrieval alone should capture most of the
  achievable gain at current scale. Flip the default only if evidence (retrieval-quality complaints,
  eval harness results) says otherwise — there's no eval harness in this repo yet either; building
  one is a prerequisite for making this call with data instead of guesswork.

## Episodic memory: wired but unused

`ChatMessageRecord` rows are written whenever `MemoryAgent(persist=True)` is active, making a full
per-session transcript available — but nothing surfaces it yet. Phase 2: an endpoint (or an intent
like "what did we discuss earlier?") that looks up a session's full history by `session_id`, distinct
from the semantic-memory fallback (which only fires on a document-retrieval miss and returns
summaries, not verbatim turns).

## Real token counting

`MemoryAgent`'s summarization token-threshold trigger currently uses a word-count proxy
(`_estimate_tokens` — `len(text.split())`), consistent with the word-count-based chunk sizing used
elsewhere in this codebase (`max_chunk_size`, `min_chunk_size`). A real tokenizer (e.g. `tiktoken` or
the provider's own tokenizer) would make the threshold meaningfully more accurate, at the cost of a
new dependency and provider-specific tokenizer selection — not pulled in speculatively.
