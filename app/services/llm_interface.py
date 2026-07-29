"""Multi-provider LLM interface — supports Gemini, OpenAI, and Anthropic."""

import json
import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from loguru import logger

from app.core.config import settings

from .models import ChatMessage, QueryType, RetrievedChunk

_DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
}


class _BaseLLM(ABC):
    """Shared logic for all LLM providers."""

    def __init__(self, intent_classifier=None):
        self.model = settings.model_name or _DEFAULT_MODELS[settings.llm_provider]
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.top_p = settings.top_p
        self._intent_classifier = intent_classifier

    @property
    def intent_classifier(self):
        if self._intent_classifier is None:
            from .intent_classifier import IntentClassifier

            self._intent_classifier = IntentClassifier()
        return self._intent_classifier

    @abstractmethod
    def _call(self, system_prompt: str, user_message: str, max_tokens: int | None = None) -> str: ...

    def generate_answer(self, query: str, retrieved_chunks: List[RetrievedChunk], intent: QueryType) -> str:
        system_prompt = self.intent_classifier.get_classification_prompt(intent)
        context = self._format_context(retrieved_chunks)
        user_message = (
            f'Based on the following document excerpts, answer this query: "{query}"\n\n'
            f"DOCUMENT EXCERPTS:\n{context}\n\n"
            f"Remember:\n"
            f"1. Answer ONLY based on the provided excerpts\n"
            f"2. If the answer is not in the excerpts, say so clearly\n"
            f"3. Be accurate and cite the source pages when relevant\n"
            f"4. For the intent type '{intent.value}', structure your answer appropriately\n"
            f"5. Be concise: default to roughly one paragraph or a short list. Only use more "
            f"space than that when the intent structurally requires it (e.g. multiple "
            f"comparison points or process steps) — never pad with restatement or extra "
            f"elaboration beyond what the question needs"
        )
        logger.info(
            f"LLM generate_answer: provider={settings.llm_provider}  model={self.model}  intent={intent.value}  context_chunks={len(retrieved_chunks)}"
        )
        logger.debug(
            f"LLM system_prompt length={len(system_prompt)}  user_message length={len(user_message)}"
        )
        start = time.time()
        try:
            result = self._call(system_prompt, user_message)
            duration = time.time() - start
            logger.info(f"LLM generate_answer OK: duration={duration:.2f}s  answer_length={len(result)}")
            logger.debug(f"LLM answer preview: {result[:200]}...")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(
                f"LLM generate_answer FAILED: duration={duration:.2f}s  error={type(e).__name__}: {e}"
            )
            raise

    def extract_claims(self, answer: str, retrieved_chunks: List[RetrievedChunk]) -> List[dict]:
        """Extract factual claims from ``answer`` and trace each one back to
        the numbered excerpt(s) (``[1]``, ``[3]``, ...) it was drawn from, so
        `SpanExtractor.extract_supporting_span` can search only within the
        LLM-indicated chunks instead of fuzzy-matching blindly across all of
        them. Returns ``[{"claim": str, "chunks": [int, ...]}, ...]`` —
        ``chunks`` is empty when a claim isn't clearly grounded in one excerpt.
        """
        system_prompt = (
            "You are an expert at extracting factual claims from text and tracing each one "
            "back to its source excerpt.\n"
            "Extract all factual claims (not opinions) from the ANSWER TEXT below.\n"
            "For each claim, identify which numbered excerpt(s) it was drawn from.\n"
            "Return ONLY a JSON array of objects with this exact shape:\n"
            '[{"claim": "Photosynthesis occurs in chloroplasts", "chunks": [1]}, '
            '{"claim": "CO2 is converted to glucose", "chunks": [1, 3]}]\n'
            "Use an empty array for `chunks` if a claim isn't clearly grounded in a specific excerpt.\n"
            'Every double quote inside a claim string MUST be escaped as \\" so the result is valid JSON — '
            "paraphrase quoted terms from the answer rather than reproducing embedded quote marks verbatim."
        )
        context = self._format_context(retrieved_chunks)
        user_message = f"DOCUMENT EXCERPTS:\n{context}\n\nANSWER TEXT:\n{answer}"
        logger.debug(f"LLM extract_claims: answer_length={len(answer)}  chunks={len(retrieved_chunks)}")
        start = time.time()
        try:
            # The chunk-indexed {"claim": ..., "chunks": [...]} shape is more
            # verbose per-claim than a plain string array, and a long
            # "comprehensive"/"explain" answer can carry a dozen-plus claims —
            # a fixed budget too close to the old plain-string-array size
            # truncates the JSON mid-array on exactly those answers.
            raw = self._call(system_prompt, user_message, max_tokens=1200)
            duration = time.time() - start
            s = raw.find("[")
            e = raw.rfind("]") + 1
            if s == -1 or e <= s:
                logger.warning(
                    f"LLM extract_claims: no JSON array found in response  duration={duration:.2f}s"
                )
                return []
            try:
                parsed = json.loads(raw[s:e])
            except json.JSONDecodeError:
                # A single malformed claim (e.g. an unescaped quote mark the
                # model didn't escape) shouldn't discard every other claim in
                # the batch — recover whatever individual objects are
                # themselves valid JSON instead of failing the whole array.
                parsed = []
                for match in re.finditer(r"\{[^{}]*\}", raw[s:e]):
                    try:
                        parsed.append(json.loads(match.group()))
                    except json.JSONDecodeError:
                        continue
                logger.warning(
                    f"LLM extract_claims: array-level JSON parse failed, recovered "
                    f"{len(parsed)} individually-valid claim object(s)"
                )
            if not isinstance(parsed, list):
                return []
            result = []
            for item in parsed:
                if isinstance(item, str):
                    # Tolerate a plain-string-array response (model deviation
                    # from the requested shape) as an unlocated claim.
                    result.append({"claim": item, "chunks": []})
                elif isinstance(item, dict) and item.get("claim"):
                    indices = [c for c in (item.get("chunks") or []) if isinstance(c, int)]
                    result.append({"claim": item["claim"], "chunks": indices})
            logger.info(f"LLM extract_claims OK: duration={duration:.2f}s  claims_found={len(result)}")
            return result
        except (json.JSONDecodeError, Exception) as exc:
            duration = time.time() - start
            logger.warning(
                f"LLM extract_claims FAILED: duration={duration:.2f}s  error={type(exc).__name__}: {exc}"
            )
            return []

    def reflect_on_answer(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        draft_answer: str,
        validator_summary: str,
        intent: QueryType,
    ) -> dict:
        """Quality-control pass over a drafted answer before it's shown to the user.

        Checks factual consistency against the retrieved excerpts, removes/hedges
        hallucinated content, and improves clarity/structure/completeness/exam
        relevance. Returns the same shape on both success and failure so callers
        never need to special-case a broken reflection pass — on any parse or
        provider failure this degrades to "no change", not an exception.
        """
        fallback = {
            "revised_answer": draft_answer,
            "materially_changed": False,
            "should_block": False,
            "issues_found": [],
        }
        system_prompt = (
            "You are a reflection and quality-control pass reviewing a draft answer for a "
            "student exam-prep assistant, before it is shown to the user.\n"
            "Check the draft against the document excerpts for:\n"
            "1. Factual consistency — does every claim hold up against the excerpts?\n"
            "2. Hallucination — remove or hedge anything not grounded in the excerpts\n"
            "3. Clarity — plain, well-organized language\n"
            "4. Structure — matches the expected shape for this intent type. The drafting "
            f"instructions were:\n{self.intent_classifier.get_classification_prompt(intent)}\n"
            "5. Completeness — uses the relevant excerpt information a student would need\n"
            "6. Exam relevance — focused on what's testable, no padding\n\n"
            "Return ONLY a JSON object with this exact shape:\n"
            '{"revised_answer": "...", "materially_changed": true, "should_block": false, '
            '"issues_found": ["..."]}\n\n'
            "Set materially_changed=true only if you changed factual content or "
            "citation-relevant wording — not for pure formatting/style polish. Set "
            "should_block=true only if the draft is fundamentally unusable (e.g. hallucinated "
            "throughout) and cannot be fixed by revision."
        )
        context = self._format_context(retrieved_chunks)
        user_message = (
            f'Original question: "{query}"\n\n'
            f"DOCUMENT EXCERPTS:\n{context}\n\n"
            f"DRAFT ANSWER:\n{draft_answer}\n\n"
            f"VALIDATOR REPORT (citation/confidence signal computed from the draft):\n{validator_summary}\n"
        )
        logger.info(f"LLM reflect_on_answer: intent={intent.value}  draft_length={len(draft_answer)}")
        start = time.time()
        try:
            raw = self._call(system_prompt, user_message, max_tokens=max(self.max_tokens, 700))
            duration = time.time() - start
            s = raw.find("{")
            e = raw.rfind("}") + 1
            if s == -1 or e <= s:
                logger.warning(
                    f"LLM reflect_on_answer: no JSON object found in response  duration={duration:.2f}s"
                )
                return fallback
            parsed = json.loads(raw[s:e])
            if not isinstance(parsed, dict) or "revised_answer" not in parsed:
                logger.warning(f"LLM reflect_on_answer: malformed JSON shape  duration={duration:.2f}s")
                return fallback
            result = {
                "revised_answer": parsed.get("revised_answer") or draft_answer,
                "materially_changed": bool(parsed.get("materially_changed", False)),
                "should_block": bool(parsed.get("should_block", False)),
                "issues_found": parsed.get("issues_found") or [],
            }
            logger.info(
                f"LLM reflect_on_answer OK: duration={duration:.2f}s  "
                f"materially_changed={result['materially_changed']}  should_block={result['should_block']}  "
                f"issues={len(result['issues_found'])}"
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
        self, query: str, retrieved_chunks: List[RetrievedChunk], intent: QueryType
    ) -> dict:
        logger.info(
            f"LLM generate_structured_answer: query={query!r}  intent={intent.value}  chunks={len(retrieved_chunks)}"
        )
        answer = self.generate_answer(query, retrieved_chunks, intent)
        claims = self.extract_claims(answer, retrieved_chunks)
        fmt = {
            QueryType.DEFINITION: "definition",
            QueryType.EXPLAIN: "comprehensive",
            QueryType.COMPARE: "comparison",
            QueryType.PROCESS: "ordered_steps",
            QueryType.EXAMPLE: "examples",
            QueryType.DIAGRAM: "description",
            QueryType.VAGUE: "general",
        }
        format_type = fmt.get(intent, "general")
        logger.info(f"LLM structured answer complete: format_type={format_type}  claims={len(claims)}")
        return {
            "answer": answer,
            "format_type": format_type,
            "claims": claims,
            "intent": intent.value,
        }

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

    def __init__(self, intent_classifier=None):
        super().__init__(intent_classifier)
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
            logger.debug(f"Gemini._call OK: duration={duration:.2f}s  response_length={len(response.text)}")
            return response.text
        except Exception as e:
            duration = time.time() - start
            logger.error(f"Gemini._call FAILED: duration={duration:.2f}s  error={type(e).__name__}: {e}")
            raise


class _OpenAILLM(_BaseLLM):
    def __init__(self, intent_classifier=None):
        super().__init__(intent_classifier)
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
                f"OpenAI._call OK: duration={duration:.2f}s  response_length={len(text)}  "
                f"tokens_in={usage.prompt_tokens if usage else '?'}  tokens_out={usage.completion_tokens if usage else '?'}"
            )
            return text
        except Exception as e:
            duration = time.time() - start
            logger.error(f"OpenAI._call FAILED: duration={duration:.2f}s  error={type(e).__name__}: {e}")
            raise


class _AnthropicLLM(_BaseLLM):
    def __init__(self, intent_classifier=None):
        super().__init__(intent_classifier)
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
            text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
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


def ClaudeInterface(intent_classifier=None) -> _BaseLLM:
    """Factory that returns the LLM backend matching ``settings.llm_provider``.

    Named ``ClaudeInterface`` for backward compatibility with the rest of the
    codebase — callers do not need to change.
    """
    provider = settings.llm_provider.lower()
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose from: {', '.join(_PROVIDERS)}")
    return cls(intent_classifier=intent_classifier)
