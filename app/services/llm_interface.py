"""Multi-provider LLM interface — supports Gemini, OpenAI, and Anthropic."""

import json
import time
from abc import ABC, abstractmethod
from typing import Optional, List
from loguru import logger

from .models import QueryType, RetrievedChunk
from app.core.config import settings

_DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
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
    def _call(self, system_prompt: str, user_message: str, max_tokens: int | None = None) -> str:
        ...

    def generate_answer(
        self, query: str, retrieved_chunks: List[RetrievedChunk], intent: QueryType
    ) -> str:
        system_prompt = self.intent_classifier.get_classification_prompt(intent)
        context = self._format_context(retrieved_chunks)
        user_message = (
            f'Based on the following document excerpts, answer this query: "{query}"\n\n'
            f"DOCUMENT EXCERPTS:\n{context}\n\n"
            f"Remember:\n"
            f"1. Answer ONLY based on the provided excerpts\n"
            f"2. If the answer is not in the excerpts, say so clearly\n"
            f"3. Be accurate and cite the source pages when relevant\n"
            f"4. For the intent type '{intent.value}', structure your answer appropriately"
        )
        logger.info(f"LLM generate_answer: provider={settings.llm_provider}  model={self.model}  intent={intent.value}  context_chunks={len(retrieved_chunks)}")
        logger.debug(f"LLM system_prompt length={len(system_prompt)}  user_message length={len(user_message)}")
        start = time.time()
        try:
            result = self._call(system_prompt, user_message)
            duration = time.time() - start
            logger.info(f"LLM generate_answer OK: duration={duration:.2f}s  answer_length={len(result)}")
            logger.debug(f"LLM answer preview: {result[:200]}...")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"LLM generate_answer FAILED: duration={duration:.2f}s  error={type(e).__name__}: {e}")
            raise

    def extract_claims(self, answer: str) -> List[str]:
        system_prompt = (
            "You are an expert at extracting factual claims from text.\n"
            "Extract all factual claims (not opinions) from the provided text.\n"
            "Return as a JSON array of strings.\n"
            'Example: ["Photosynthesis occurs in chloroplasts", "CO2 is converted to glucose"]'
        )
        logger.debug(f"LLM extract_claims: answer_length={len(answer)}")
        start = time.time()
        try:
            raw = self._call(system_prompt, f"Extract claims: {answer}", max_tokens=500)
            duration = time.time() - start
            s = raw.find("[")
            e = raw.rfind("]") + 1
            if s != -1 and e > s:
                claims = json.loads(raw[s:e])
                result = claims if isinstance(claims, list) else []
                logger.info(f"LLM extract_claims OK: duration={duration:.2f}s  claims_found={len(result)}")
                return result
            logger.warning(f"LLM extract_claims: no JSON array found in response  duration={duration:.2f}s")
            return []
        except (json.JSONDecodeError, Exception) as exc:
            duration = time.time() - start
            logger.warning(f"LLM extract_claims FAILED: duration={duration:.2f}s  error={type(exc).__name__}: {exc}")
            return []

    def generate_structured_answer(
        self, query: str, retrieved_chunks: List[RetrievedChunk], intent: QueryType
    ) -> dict:
        logger.info(f"LLM generate_structured_answer: query={query!r}  intent={intent.value}  chunks={len(retrieved_chunks)}")
        answer = self.generate_answer(query, retrieved_chunks, intent)
        claims = self.extract_claims(answer)
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
    def __init__(self, intent_classifier=None):
        super().__init__(intent_classifier)
        import google.generativeai as genai

        api_key = settings.gemini_api_key
        logger.debug(f"Gemini init: api_key={'set (' + api_key[:8] + '...)' if api_key else 'NOT SET'}")
        genai.configure(api_key=api_key)
        self._genai = genai
        logger.info(f"LLM provider: Gemini ({self.model})")

    def _call(self, system_prompt: str, user_message: str, max_tokens: int | None = None) -> str:
        logger.debug(f"Gemini._call: model={self.model}  max_tokens={max_tokens or self.max_tokens}  temp={self.temperature}")
        start = time.time()
        try:
            model = self._genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
                generation_config=self._genai.types.GenerationConfig(
                    max_output_tokens=max_tokens or self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                ),
            )
            response = model.generate_content(user_message)
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
        logger.debug(f"OpenAI init: api_key={'set (' + api_key[:8] + '...)' if api_key else 'NOT SET'}")
        self._client = OpenAI(api_key=api_key)
        logger.info(f"LLM provider: OpenAI ({self.model})")

    def _call(self, system_prompt: str, user_message: str, max_tokens: int | None = None) -> str:
        logger.debug(f"OpenAI._call: model={self.model}  max_tokens={max_tokens or self.max_tokens}  temp={self.temperature}")
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
        logger.debug(f"Anthropic init: api_key={'set (' + api_key[:8] + '...)' if api_key else 'NOT SET'}")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._anthropic = anthropic
        logger.info(f"LLM provider: Anthropic ({self.model})")

    def _call(self, system_prompt: str, user_message: str, max_tokens: int | None = None) -> str:
        logger.debug(f"Anthropic._call: model={self.model}  max_tokens={max_tokens or self.max_tokens}  temp={self.temperature}")
        start = time.time()
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            duration = time.time() - start
            text = response.content[0].text
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
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Choose from: {', '.join(_PROVIDERS)}"
        )
    return cls(intent_classifier=intent_classifier)
