"""MemoryAgent — records conversation turns into short-term chat history,
with opt-in persistence into the three memory tiers described in the
architecture plan (short-term/episodic via ChatSession/ChatMessageRecord,
long-term/semantic via ConversationMemory + the shared FAISS index).

``persist`` defaults to False — exactly today's pure in-memory behavior, so
existing callers/tests are unaffected. Setting it to True is forward-compat
plumbing for a real session concept (there is no per-user session today,
just the single shared ``ExamPrepBot.chat_history`` — see architecture plan);
nothing in this phase turns it on by default.
"""

import uuid
from datetime import datetime
from typing import Callable, List, Optional

from loguru import logger

from app.core.config import settings
from app.core.conversation_sessions import get_or_create_chat_session
from app.models.db_models import ChatMessageRecord, ChatSession, ConversationMemory

from ..models import ChatMessage, QueryType


def _estimate_tokens(text: str) -> int:
    """Word-count proxy for token count — no tokenizer dependency wired yet,
    consistent with the word-count-based chunk sizing used elsewhere."""
    return len(text.split())


def _derive_title(text: str, limit: int = 50) -> str:
    """Conversation title derived from the first user message — a plain
    truncation, no extra LLM call. Authoritative once stored; user-renameable
    afterward via the conversations API."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


class MemoryAgent:
    """Wraps a bot's `chat_history` list, stored by reference (see `RetrievalAgent`)."""

    def __init__(
        self,
        chat_history: List[ChatMessage],
        persist: bool = False,
        session_id: Optional[str] = None,
        db_session_factory: Optional[Callable] = None,
        llm=None,
        vector_store_manager=None,
    ):
        self.chat_history = chat_history
        self.persist = persist
        self.session_id = session_id
        self.db_session_factory = db_session_factory
        self.llm = llm
        self.vector_store_manager = vector_store_manager

    def record_turn(
        self,
        query: str,
        answer: str,
        intent: Optional[QueryType] = None,
        format_type: Optional[str] = None,
        session_id: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> None:
        now = datetime.now().isoformat()
        self.chat_history.append(ChatMessage(role="user", content=query, timestamp=now, intent_type=intent))
        self.chat_history.append(
            ChatMessage(role="assistant", content=answer, timestamp=now, intent_type=intent)
        )

        if self.persist:
            # session_id/device_id are per-call parameters, not mutated instance
            # state: this agent is a process-wide singleton shared across
            # concurrent requests (thread-pool-executed), so storing "the
            # current conversation" on self would race between requests for
            # different conversations. self.session_id is only a fallback
            # default, kept for backward compatibility with direct construction.
            resolved_session_id = session_id or self.session_id
            self._persist_turn(query, answer, intent, format_type, resolved_session_id, device_id)

    def get_recent_turns(self, session_id: Optional[str], max_messages: int = 6) -> List[ChatMessage]:
        """Return the last ``max_messages`` (user+assistant combined) for this
        conversation, oldest-first, to thread into the drafting/reflection
        prompts as short-term context. Returns ``[]`` when persistence isn't
        configured or there's no session_id — ``self.chat_history`` is a
        single list shared across every conversation this process has ever
        handled (not scoped by session), so it's not a safe fallback source
        here; without a real session to query, there's no reliable per-
        conversation history to draw from at all."""
        if not (self.persist and session_id and self.db_session_factory):
            return []
        db = None
        try:
            db = self.db_session_factory()
            rows = (
                db.query(ChatMessageRecord)
                .filter_by(session_id=session_id)
                .order_by(ChatMessageRecord.id.desc())
                .limit(max_messages)
                .all()
            )
            rows.reverse()
            return [
                ChatMessage(
                    role=r.role,
                    content=r.content,
                    timestamp=r.created_at.isoformat() if r.created_at else "",
                    intent_type=QueryType(r.intent) if r.intent else None,
                )
                for r in rows
            ]
        except Exception as exc:
            logger.warning(f"[MEMORY] Failed to fetch recent turns for context: {type(exc).__name__}: {exc}")
            return []
        finally:
            if db is not None:
                db.close()

    def _persist_turn(
        self,
        query: str,
        answer: str,
        intent: Optional[QueryType],
        format_type: Optional[str],
        session_id: Optional[str],
        device_id: Optional[str] = None,
    ) -> None:
        if not (session_id and self.db_session_factory):
            logger.warning(
                "[MEMORY] persist=True but session_id/db_session_factory missing — skipping persistence"
            )
            return

        db = self.db_session_factory()
        try:
            session_row = get_or_create_chat_session(
                db, session_id, device_id=device_id, title=_derive_title(query)
            )

            last_msg = (
                db.query(ChatMessageRecord)
                .filter_by(session_id=session_id)
                .order_by(ChatMessageRecord.id.desc())
                .first()
            )
            running_total = last_msg.token_count if last_msg and last_msg.token_count else 0
            intent_value = intent.value if intent else None

            user_tokens = running_total + _estimate_tokens(query)
            db.add(
                ChatMessageRecord(
                    session_id=session_id,
                    role="user",
                    content=query,
                    intent=intent_value,
                    token_count=user_tokens,
                )
            )
            db.flush()

            assistant_tokens = user_tokens + _estimate_tokens(answer)
            assistant_msg = ChatMessageRecord(
                session_id=session_id,
                role="assistant",
                content=answer,
                intent=intent_value,
                format_type=format_type,
                token_count=assistant_tokens,
            )
            db.add(assistant_msg)
            session_row.turn_count += 1  # type: ignore[assignment]
            # The only place last_activity_at should move — a real turn, not
            # a title edit (see ChatSession.last_activity_at's comment and
            # list_conversations' order_by).
            session_row.last_activity_at = datetime.now()  # type: ignore[assignment]
            db.flush()

            self._maybe_summarize(db, session_row, assistant_msg, session_id)
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.warning(
                f"[MEMORY] Persistence failed, turn kept in-memory only: {type(exc).__name__}: {exc}"
            )
        finally:
            db.close()

    def _maybe_summarize(
        self, db, session_row: ChatSession, latest_message: ChatMessageRecord, session_id: str
    ) -> None:
        """Dual threshold — whichever fires first: every N turns, or the
        cumulative (running-total) token count since the last summary
        crossing a threshold. Both checks are O(1) column reads/lookups,
        never a COUNT/SUM scan over the full session history."""
        turn_trigger = session_row.turn_count % settings.memory_summarize_every_n_turns == 0

        baseline_tokens = 0
        if session_row.last_summarized_message_id:
            baseline_msg = db.get(ChatMessageRecord, session_row.last_summarized_message_id)
            baseline_tokens = baseline_msg.token_count if baseline_msg and baseline_msg.token_count else 0
        token_trigger = (
            latest_message.token_count or 0
        ) - baseline_tokens >= settings.memory_summarize_token_threshold

        if not (turn_trigger or token_trigger):
            return
        if self.llm is None:
            logger.debug("[MEMORY] Summarization triggered but no llm configured — skipping this round")
            return

        from_id = (session_row.last_summarized_message_id or 0) + 1
        turns = (
            db.query(ChatMessageRecord)
            .filter(ChatMessageRecord.session_id == session_id, ChatMessageRecord.id >= from_id)
            .order_by(ChatMessageRecord.id)
            .all()
        )
        if not turns:
            return

        transcript = [
            ChatMessage(
                role=t.role,
                content=t.content,
                timestamp=t.created_at.isoformat() if t.created_at else "",
            )
            for t in turns
        ]
        summary_text = self.llm.summarize_conversation(transcript)
        if not summary_text:
            logger.debug("[MEMORY] Summarization produced no text — skipping this round")
            return

        memory_id = str(uuid.uuid4())
        memory_row = ConversationMemory(
            id=memory_id,
            session_id=session_id,
            summary_text=summary_text,
            covers_from_message_id=turns[0].id,
            covers_to_message_id=turns[-1].id,
            embedded=False,
        )
        db.add(memory_row)
        session_row.last_summarized_message_id = turns[-1].id
        db.flush()

        if self.vector_store_manager is not None:
            try:
                self.vector_store_manager.add_memory(summary_text, session_id=session_id, memory_id=memory_id)
                memory_row.embedded = True  # type: ignore[assignment]
            except Exception as exc:
                logger.warning(
                    f"[MEMORY] Embedding summary failed — row kept with embedded=False for retry: "
                    f"{type(exc).__name__}: {exc}"
                )
