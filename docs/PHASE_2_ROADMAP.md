# Phase 2 Roadmap

Phase 1 (this rebuild) delivered the multi-agent pipeline foundation: Orchestrator/Retrieval/
Knowledge/Reflection/Memory agents, hybrid (dense + BM25) retrieval with opt-in cross-encoder
reranking, structural chunking, chunk-indexed citations, and an opt-in conversation-memory
persistence layer. Everything below was scoped out of Phase 1 by explicit decision, not because it's
unimportant — this is a punch list for the next phase, not a wishlist.

## Deferred agents

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

## Multi-tenancy: architecture-ready, not built

Phase 1 made the schema forward-compatible without building the feature:
- `ChatSession.user_id` and `ConversationMemory.user_id` are nullable columns, added specifically so
  real multi-user auth won't require another migration.
- Today's actual auth model is unchanged: a single shared `APP_API_KEY` via `X-API-Key` (or
  `?api_key=` for WebSocket) — see `app/core/security.py`.

Phase 2 work: real login/signup, per-user API keys or JWT/session auth, and threading a real
`user_id`/`session_id` through `get_bot()` (today's `ExamPrepBot` singleton has one shared
`chat_history` — there's no per-user isolation to remove, since there's no per-user concept yet).

## Progress tracking

Depends on the Quiz/Flashcard agents existing first (there's nothing to track progress *on* yet
otherwise) — e.g. mastery-per-topic derived from quiz results, spaced-repetition scheduling state,
and a dashboard surfacing it. Revisit once those agents land.

## Retrieval/embedding follow-ups (reassess, don't build speculatively)

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
