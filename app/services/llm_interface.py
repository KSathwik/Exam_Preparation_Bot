"""Multi-provider LLM interface — supports Gemini, OpenAI, and Anthropic."""

import json
import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from loguru import logger

from app.core.config import redact_query_for_log, settings

from .models import ChatMessage, QueryType, RetrievedChunk
from .response_formats import LENGTH_MAX_TOKENS, RESPONSE_FORMAT_TEMPLATES, ResponseFormat

_DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
}


class _BaseLLM(ABC):
    """Shared logic for all LLM providers."""

    def __init__(self):
        self.model = settings.model_name or _DEFAULT_MODELS[settings.llm_provider]
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.top_p = settings.top_p

    @abstractmethod
    def _call(self, system_prompt: str, user_message: str, max_tokens: int | None = None) -> str: ...

    def generate_answer_with_claims(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        intent: QueryType,
        response_format: ResponseFormat = ResponseFormat.GENERAL,
        history: Optional[List[ChatMessage]] = None,
    ) -> dict:
        """Draft an answer AND trace its factual claims back to the numbered
        excerpt(s) (``[1]``, ``[3]``, ...) they were drawn from, in a single
        call — merged from two separate calls (draft, then a dedicated
        claim-extraction pass) to cut a full LLM round trip off every turn's
        time-to-first-token. ``claims`` feeds `SpanExtractor.extract_supporting_span`
        so it can search only the LLM-indicated chunks instead of fuzzy-matching
        blindly across all of them.

        ``response_format`` drives the answer's *structure* (headings, bullets,
        a table, flashcards, ...) — this is the presentation axis, orthogonal
        to ``intent`` which only still matters here as a fallback label; see
        ``response_formats.py`` for the actual per-format instructions.

        Returns ``{"answer": str, "claims": [{"claim": str, "chunks": [int, ...]}, ...]}``.
        A response that fails to parse as the requested JSON object degrades
        gracefully (see the three-tier fallback below) rather than losing the
        answer text entirely — a formatting slip must never cost the user
        their answer.
        """
        template = RESPONSE_FORMAT_TEMPLATES[response_format]
        system_prompt = (
            "You are a helpful study assistant for a student preparing for exams.\n"
            f"{template.prompt_instructions}\n\n"
            "Respond with ONLY a JSON object of this exact shape:\n"
            '{"answer": "<your answer as a plain string>", '
            '"claims": [{"claim": "...", "chunks": [1]}, {"claim": "...", "chunks": [1, 3]}]}\n'
            '"claims" traces every factual claim in "answer" back to the numbered excerpt(s) '
            "it was drawn from — use an empty array for `chunks` if a claim isn't clearly "
            "grounded in a specific excerpt.\n"
            'Every double quote inside "answer" or a claim string MUST be escaped as \\" so the '
            "whole response is valid JSON — paraphrase quoted terms rather than reproducing "
            'embedded quote marks verbatim. Newlines inside "answer" (e.g. between Markdown '
            "headings/list items) MUST be escaped as \\n, not literal line breaks.\n"
            "If a RECENT CONVERSATION is provided below, use it only to resolve what the current "
            'question is referring to (pronouns like "it"/"that", or "the second one") — never '
            "as a source of facts. Every factual claim must still come from the document content.\n"
            'In "answer", refer to the source material as "the document" or "your uploaded '
            'material" — never as "excerpts" or "chunks", which are internal terms a student '
            "wouldn't recognize."
        )
        context = self._format_context(retrieved_chunks)
        user_message = (
            f"{self._format_history(history)}"
            f'Based on the following material from the uploaded document(s), answer this query: "{query}"\n\n'
            f"DOCUMENT CONTENT:\n{context}\n\n"
            f'Remember, for the "answer" field:\n'
            f"1. Answer ONLY based on the provided material\n"
            f"2. If the answer is not in the document, say so clearly\n"
            f"3. Be accurate and cite the source pages when relevant\n"
            f"4. Follow the structure instructions above exactly — do not fall back to plain "
            f"prose paragraphs if a specific structure (list, table, flashcards, etc.) was requested\n"
            f"5. Never pad with restatement or extra elaboration beyond what the question needs"
        )
        max_tokens = LENGTH_MAX_TOKENS[template.default_length]
        logger.info(
            f"LLM generate_answer_with_claims: provider={settings.llm_provider}  model={self.model}  "
            f"intent={intent.value}  response_format={response_format.value}  "
            f"context_chunks={len(retrieved_chunks)}"
        )
        start = time.time()
        try:
            # +500 over the plain-answer budget: this call must also carry
            # the claims array and JSON-envelope/escaping overhead — reusing
            # the bare answer ceiling risks truncating the JSON exactly on
            # long "comprehensive"/"explain" answers with a dozen-plus claims.
            raw = self._call(system_prompt, user_message, max_tokens=max_tokens + 500)
            duration = time.time() - start
            result = self._parse_answer_with_claims(raw)
            # Normalize result to concrete types so callers and loggers don't
            # need to guard against None or unexpected shapes.
            answer = result.get("answer") or ""
            claims = result.get("claims") or []
            logger.info(
                f"LLM generate_answer_with_claims OK: duration={duration:.2f}s  "
                f"answer_length={len(answer)}  claims_found={len(claims)}"
            )
            return {"answer": answer, "claims": claims}
        except Exception as e:
            duration = time.time() - start
            logger.error(
                f"LLM generate_answer_with_claims FAILED: duration={duration:.2f}s  "
                f"error={type(e).__name__}: {e}"
            )
            raise

    @staticmethod
    def _escape_raw_newlines_in_json_strings(text: str) -> str:
        """A model asked for multi-paragraph/heading-heavy Markdown inside a
        JSON string field doesn't reliably escape *every* embedded newline as
        ``\\n`` despite being told to — the more structured the requested
        format (headings, lists), the more paragraph breaks there are to get
        right. A single missed one is a raw control character, which is
        invalid inside a JSON string literal and fails ``json.loads``
        entirely (not just for that field — the whole object). Repairs this
        mechanically by walking the text and escaping newlines only when
        actually inside a string literal (tracked via unescaped-quote
        toggling), never ones that are just structural whitespace between
        JSON tokens."""
        out = []
        in_string = False
        escaped = False
        for ch in text:
            if in_string:
                if escaped:
                    out.append(ch)
                    escaped = False
                elif ch == "\\":
                    out.append(ch)
                    escaped = True
                elif ch == '"':
                    out.append(ch)
                    in_string = False
                elif ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    pass  # a preceding \r\n already became \\n above; a bare \r is dropped
                else:
                    out.append(ch)
            else:
                if ch == '"':
                    in_string = True
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _parse_answer_with_claims(raw: str) -> dict:
        s, e = raw.find("{"), raw.rfind("}") + 1
        # A response that hit max_tokens mid-generation (a genuinely verbose
        # MCQ/exam-questions answer, say) never emits a closing brace at all —
        # raw.rfind("}") then returns -1, and requiring e > s here would skip
        # every recovery tier below entirely, even though there's a perfectly
        # readable partial answer sitting right after the opening brace. Fall
        # back to "rest of the text from the opening brace" so at least the
        # best-effort extraction below still gets a chance.
        if s != -1 and e <= s:
            e = len(raw)
        if s != -1 and e > s:
            repaired = _BaseLLM._escape_raw_newlines_in_json_strings(raw[s:e])
            try:
                parsed = json.loads(repaired)
                answer = parsed.get("answer")
                if isinstance(answer, str) and answer.strip():
                    return {"answer": answer, "claims": _BaseLLM._normalize_claims(parsed.get("claims"))}
            except json.JSONDecodeError:
                pass
            # The whole object didn't parse (e.g. an unescaped quote inside a
            # claim) — the answer field alone may still be well-formed, so
            # recover just that rather than losing the answer over an
            # unrelated claims-array formatting slip.
            match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', repaired, re.DOTALL)
            if match:
                try:
                    recovered = json.loads(f'"{match.group(1)}"')
                    logger.warning(
                        "LLM generate_answer_with_claims: recovered answer field from malformed JSON"
                    )
                    return {"answer": recovered, "claims": []}
                except json.JSONDecodeError:
                    pass
            # Both recovery attempts above assume the answer text itself has
            # no unescaped quote in it — a claim-heavy, list/table-heavy
            # answer (MCQs, comparisons, ...) occasionally has exactly that
            # (e.g. quoting a term), which stops the regex's own quote
            # matching short and both parses above fail on the truncated
            # fragment. Last resort: locate the "answer" field by its
            # structural boundary with "claims" instead of by quote-matching,
            # and hand-unescape the common sequences — never strictly
            # validated JSON, but strictly better than showing the user a
            # literal JSON envelope (braces, escaped quotes, visible \n).
            best_effort = _BaseLLM._best_effort_answer_extraction(repaired)
            if best_effort:
                logger.warning(
                    "LLM generate_answer_with_claims: best-effort answer extraction from malformed JSON"
                )
                return {"answer": best_effort, "claims": []}
        logger.warning(
            "LLM generate_answer_with_claims: no valid JSON found, using raw response as the answer"
        )
        return {"answer": raw.strip(), "claims": []}

    @staticmethod
    def _best_effort_answer_extraction(text: str) -> Optional[str]:
        """Structural (not quote-matching) recovery of the "answer" field's
        raw text — the fallback of last resort when even the tolerant
        regex-based recovery above fails. That regex stops at the first
        unescaped quote it finds, so it can't tell a genuine field-closing
        quote from one embedded partway through the answer's own text (a
        quoted term, an apostrophe the model escaped JS/Python-style as
        ``\\'`` — valid in those languages, not in JSON, so even the
        recovered fragment's own ``json.loads`` rejects it). Anchoring on the
        structural ``", "claims"`` boundary instead survives both cases."""
        match = re.search(r'"answer"\s*:\s*"', text)
        if not match:
            return None
        start = match.end()
        end_match = re.search(r'"\s*,\s*"claims"', text[start:])
        end = start + end_match.start() if end_match else len(text)
        fragment = text[start:end]
        if not fragment.strip():
            return None
        return (
            fragment.replace('\\"', '"')
            .replace("\\'", "'")
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\\\", "\\")
            .strip()
        )

    @staticmethod
    def _normalize_claims(raw_claims) -> List[dict]:
        """Tolerant claim-list normalization shared by ``generate_answer_with_claims``
        and ``reflect_on_answer`` — accepts the requested
        ``{"claim": ..., "chunks": [...]}`` shape, a plain string (model
        deviation, treated as an unlocated claim), and filters ``chunks`` to
        ints only."""
        if not isinstance(raw_claims, list):
            return []
        result = []
        for item in raw_claims:
            if isinstance(item, str):
                result.append({"claim": item, "chunks": []})
            elif isinstance(item, dict) and item.get("claim"):
                indices = [c for c in (item.get("chunks") or []) if isinstance(c, int)]
                result.append({"claim": item["claim"], "chunks": indices})  # type: ignore[dict-item]
        return result

    def reflect_on_answer(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        draft_answer: str,
        validator_summary: str,
        intent: QueryType,
        response_format: ResponseFormat = ResponseFormat.GENERAL,
        history: Optional[List[ChatMessage]] = None,
    ) -> dict:
        """Quality-control pass over a drafted answer before it's shown to the user.

        Checks factual consistency against the retrieved excerpts, removes/hedges
        hallucinated content, and improves clarity/structure/completeness/exam
        relevance. Also re-traces claims for the (possibly revised) answer in
        this same call — merged in so a materially-changed answer never needs
        a separate re-extraction round trip. Returns the same shape on both
        success and failure so callers never need to special-case a broken
        reflection pass — on any parse or provider failure this degrades to
        "no change", not an exception. ``claims`` is ``None`` on failure,
        signaling callers to reuse the draft's own claims instead.
        """
        fallback = {
            "revised_answer": draft_answer,
            "materially_changed": False,
            "should_block": False,
            "issues_found": [],
            "claims": None,
        }
        system_prompt = (
            "You are a reflection and quality-control pass reviewing a draft answer for a "
            "student exam-prep assistant, before it is shown to the user.\n"
            "Check the draft against the document content for:\n"
            "1. Factual consistency — does every claim hold up against the source content?\n"
            "2. Hallucination — remove or hedge anything not grounded in the source content\n"
            "3. Clarity — plain, well-organized language\n"
            "4. Structure — matches the expected shape. The drafting instructions were: "
            f"{RESPONSE_FORMAT_TEMPLATES[response_format].structure_note}\n"
            "5. Completeness — uses the relevant document information a student would need\n"
            "6. Exam relevance — focused on what's testable, no padding\n\n"
            "Return ONLY a JSON object with this exact shape:\n"
            '{"revised_answer": "...", "materially_changed": true, "should_block": false, '
            '"issues_found": ["..."], "claims": [{"claim": "...", "chunks": [1]}]}\n\n'
            "Set materially_changed=true only if you changed factual content or "
            "citation-relevant wording — not for pure formatting/style polish. Set "
            "should_block=true only if the draft is fundamentally unusable (e.g. hallucinated "
            'throughout) and cannot be fixed by revision. "claims" traces every factual claim '
            'in "revised_answer" back to the numbered source excerpt(s) it was drawn from — same '
            "shape and rules as the draft's own claims; empty array for `chunks` if a claim "
            'isn\'t clearly grounded in a specific source excerpt. In "revised_answer", refer to '
            'the source material as "the document" or "your uploaded material" — never as '
            '"excerpts" or "chunks", which are internal terms a student wouldn\'t recognize.'
        )
        context = self._format_context(retrieved_chunks)
        user_message = (
            f"{self._format_history(history)}"
            f'Original question: "{query}"\n\n'
            f"DOCUMENT CONTENT:\n{context}\n\n"
            f"DRAFT ANSWER:\n{draft_answer}\n\n"
            f"VALIDATOR REPORT (citation/confidence signal computed from the draft):\n{validator_summary}\n"
        )
        logger.info(
            f"LLM reflect_on_answer: intent={intent.value}  response_format={response_format.value}  "
            f"draft_length={len(draft_answer)}"
        )
        start = time.time()
        try:
            # +500 over the draft's own budget (itself derived from
            # response_format's length tier): this call must reproduce the
            # full (possibly near-max-length) draft inside a JSON wrapper —
            # escaping expansion plus materially_changed/should_block/
            # issues_found overhead — reusing the exact same ceiling risks
            # truncating the JSON even when the draft itself fit comfortably.
            max_tokens = LENGTH_MAX_TOKENS[RESPONSE_FORMAT_TEMPLATES[response_format].default_length]
            raw = self._call(system_prompt, user_message, max_tokens=max_tokens + 500)
            duration = time.time() - start
            s = raw.find("{")
            e = raw.rfind("}") + 1
            if s == -1 or e <= s:
                logger.warning(
                    f"LLM reflect_on_answer: no JSON object found in response  duration={duration:.2f}s"
                )
                return fallback
            parsed = json.loads(_BaseLLM._escape_raw_newlines_in_json_strings(raw[s:e]))
            if not isinstance(parsed, dict) or "revised_answer" not in parsed:
                logger.warning(f"LLM reflect_on_answer: malformed JSON shape  duration={duration:.2f}s")
                return fallback
            result = {
                "revised_answer": parsed.get("revised_answer") or draft_answer,
                "materially_changed": bool(parsed.get("materially_changed", False)),
                "should_block": bool(parsed.get("should_block", False)),
                "issues_found": parsed.get("issues_found") or [],
                "claims": self._normalize_claims(parsed.get("claims")),
            }
            logger.info(
                f"LLM reflect_on_answer OK: duration={duration:.2f}s  "
                f"materially_changed={result['materially_changed']}  should_block={result['should_block']}  "
                f"issues={len(result['issues_found'])}  claims={len(result['claims'])}"  # type: ignore[arg-type]
            )
            return result
        except (json.JSONDecodeError, Exception) as exc:
            duration = time.time() - start
            logger.warning(
                f"LLM reflect_on_answer FAILED: duration={duration:.2f}s  error={type(exc).__name__}: {exc}"
            )
            return fallback

    def summarize_conversation(self, turns: List[ChatMessage]) -> str:
        """Summarize a slice of conversation turns into a compact memory for later
        semantic retrieval. Returns "" on failure rather than raising — callers
        should treat an empty summary as "skip this summarization round", not crash."""
        system_prompt = (
            "Summarize the key topics, terms, and conclusions from this conversation excerpt "
            "in under 150 words, preserving specific terminology the student asked about, for "
            "later retrieval as a memory of this discussion."
        )
        transcript = "\n".join(f"{t.role}: {t.content}" for t in turns)
        logger.debug(f"LLM summarize_conversation: turns={len(turns)}  transcript_length={len(transcript)}")
        start = time.time()
        try:
            summary = self._call(system_prompt, transcript, max_tokens=300)
            duration = time.time() - start
            logger.info(
                f"LLM summarize_conversation OK: duration={duration:.2f}s  summary_length={len(summary)}"
            )
            return summary.strip()
        except Exception as exc:
            duration = time.time() - start
            logger.warning(
                f"LLM summarize_conversation FAILED: duration={duration:.2f}s  error={type(exc).__name__}: {exc}"
            )
            return ""

    def generate_structured_answer(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        intent: QueryType,
        response_format: ResponseFormat = ResponseFormat.GENERAL,
        history: Optional[List[ChatMessage]] = None,
    ) -> dict:
        logger.info(
            f"LLM generate_structured_answer: query={redact_query_for_log(query)}  intent={intent.value}  "
            f"response_format={response_format.value}  chunks={len(retrieved_chunks)}"
        )
        drafted = self.generate_answer_with_claims(
            query, retrieved_chunks, intent, response_format=response_format, history=history
        )
        answer = drafted["answer"]
        claims = drafted["claims"]
        # format_type is simply the classified response_format's own value now
        # — no more a hand-maintained intent->label dict, since presentation
        # shape is response_format's whole job (see response_formats.py).
        format_type = response_format.value
        logger.info(f"LLM structured answer complete: format_type={format_type}  claims={len(claims)}")
        return {
            "answer": answer,
            "format_type": format_type,
            "claims": claims,
            "intent": intent.value,
        }

    @staticmethod
    def _format_history(history: Optional[List[ChatMessage]]) -> str:
        """Recent conversation turns, formatted as a prompt block — empty
        string when there's no history, so callers can splice this in
        unconditionally without an extra branch. Kept short (callers already
        cap how many turns are fetched) since this is purely for resolving
        references in the CURRENT question, not a source of facts."""
        if not history:
            return ""
        transcript = "\n".join(f"{m.role}: {m.content}" for m in history)
        return (
            f"RECENT CONVERSATION (for reference resolution only, not a source of facts):\n{transcript}\n\n"
        )

    @staticmethod
    def _format_context(chunks: List[RetrievedChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            page = chunk.metadata.page_number
            section = chunk.metadata.section_title or "Unknown"
            parts.append(
                f"[{i}] (Page {page}, Section: {section}, "
                f"Relevance: {chunk.relevance_score:.2f})\n{chunk.content}\n"
            )
        return "\n".join(parts)


# ── Provider implementations ──────────────────────────────────────────


class _GeminiLLM(_BaseLLM):
    """Uses ``google-genai`` — the ``google-generativeai`` package it replaces
    reached end-of-life and receives no further updates or security fixes."""

    def __init__(self):
        super().__init__()
        from google import genai

        api_key = settings.gemini_api_key
        logger.debug(f"Gemini init: api_key={'set' if api_key else 'NOT SET'}")
        self._client = genai.Client(api_key=api_key)
        logger.info(f"LLM provider: Gemini ({self.model})")

    def _call(self, system_prompt: str, user_message: str, max_tokens: int | None = None) -> str:
        from google.genai import types

        logger.debug(
            f"Gemini._call: model={self.model}  max_tokens={max_tokens or self.max_tokens}  temp={self.temperature}"
        )
        start = time.time()
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens or self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                ),
            )
            duration = time.time() - start
            logger.debug(f"Gemini._call OK: duration={duration:.2f}s  response_length={len(response.text)}")  # type: ignore[arg-type]
            return response.text  # type: ignore[return-value]
        except Exception as e:
            duration = time.time() - start
            logger.error(f"Gemini._call FAILED: duration={duration:.2f}s  error={type(e).__name__}: {e}")
            raise


class _OpenAILLM(_BaseLLM):
    def __init__(self):
        super().__init__()
        from openai import OpenAI

        api_key = settings.openai_api_key
        logger.debug(f"OpenAI init: api_key={'set' if api_key else 'NOT SET'}")
        self._client = OpenAI(api_key=api_key)
        logger.info(f"LLM provider: OpenAI ({self.model})")

    def _call(self, system_prompt: str, user_message: str, max_tokens: int | None = None) -> str:
        logger.debug(
            f"OpenAI._call: model={self.model}  max_tokens={max_tokens or self.max_tokens}  temp={self.temperature}"
        )
        start = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            duration = time.time() - start
            text = response.choices[0].message.content
            usage = response.usage
            logger.debug(
                f"OpenAI._call OK: duration={duration:.2f}s  response_length={len(text)}  "  # type: ignore[arg-type]
                f"tokens_in={usage.prompt_tokens if usage else '?'}  tokens_out={usage.completion_tokens if usage else '?'}"
            )
            return text  # type: ignore[return-value]
        except Exception as e:
            duration = time.time() - start
            logger.error(f"OpenAI._call FAILED: duration={duration:.2f}s  error={type(e).__name__}: {e}")
            raise


class _AnthropicLLM(_BaseLLM):
    def __init__(self):
        super().__init__()
        import anthropic

        api_key = settings.anthropic_api_key
        logger.debug(f"Anthropic init: api_key={'set' if api_key else 'NOT SET'}")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._anthropic = anthropic
        logger.info(f"LLM provider: Anthropic ({self.model})")

    def _call(self, system_prompt: str, user_message: str, max_tokens: int | None = None) -> str:
        logger.debug(f"Anthropic._call: model={self.model}  max_tokens={max_tokens or self.max_tokens}")
        start = time.time()
        try:
            # Current-generation Claude models (Opus 5/4.8/4.7, Sonnet 5, Fable 5) reject
            # temperature/top_p/top_k outright (400) rather than just deprecating them —
            # so these aren't forwarded here. Steer behavior via prompting instead.
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            duration = time.time() - start
            # response.content[0] isn't reliably the text block — current-generation
            # models can emit a ThinkingBlock (or other non-text block) first, so pick
            # out the text block(s) explicitly rather than indexing blindly.
            # Some block types in Anthropic responses may not have a `.text`
            # attribute; use getattr to safely extract text where present.
            text_parts = [
                getattr(block, "text", "")
                for block in response.content
                if getattr(block, "type", None) == "text"
            ]
            text = "".join(text_parts)
            logger.debug(
                f"Anthropic._call OK: duration={duration:.2f}s  response_length={len(text)}  "
                f"tokens_in={response.usage.input_tokens}  tokens_out={response.usage.output_tokens}"
            )
            return text
        except Exception as e:
            duration = time.time() - start
            logger.error(f"Anthropic._call FAILED: duration={duration:.2f}s  error={type(e).__name__}: {e}")
            raise


# ── Factory ───────────────────────────────────────────────────────────

_PROVIDERS = {
    "gemini": _GeminiLLM,
    "openai": _OpenAILLM,
    "anthropic": _AnthropicLLM,
}


def ClaudeInterface() -> _BaseLLM:
    """Factory that returns the LLM backend matching ``settings.llm_provider``.

    Named ``ClaudeInterface`` for backward compatibility with the rest of the
    codebase — callers do not need to change.
    """
    provider = settings.llm_provider.lower()
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose from: {', '.join(_PROVIDERS)}")
    # mypy sometimes erroneously flags this instantiation as abstract;
    # silence that false positive — at runtime `cls` is a concrete subclass.
    return cls()  # type: ignore[abstract]
