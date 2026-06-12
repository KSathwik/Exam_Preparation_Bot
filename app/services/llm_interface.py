"""Anthropic Claude API interface."""

import json
from typing import Optional, List
from loguru import logger
import anthropic
from .models import QueryType, RetrievedChunk
from app.core.config import settings


class ClaudeInterface:
    """Interface for interacting with Claude API."""

    def __init__(self, intent_classifier=None):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.model_name
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
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def extract_claims(self, answer: str) -> List[str]:
        system_prompt = (
            "You are an expert at extracting factual claims from text.\n"
            "Extract all factual claims (not opinions) from the provided text.\n"
            "Return as a JSON array of strings.\n"
            'Example: ["Photosynthesis occurs in chloroplasts", "CO2 is converted to glucose"]'
        )
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": f"Extract claims: {answer}"}],
            )
            claims = json.loads(response.content[0].text)
            return claims if isinstance(claims, list) else []
        except (json.JSONDecodeError, anthropic.APIError) as e:
            logger.warning(f"Claim extraction failed: {e}")
            return []

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            page = chunk.metadata.page_number
            section = chunk.metadata.section_title or "Unknown"
            parts.append(
                f"[{i}] (Page {page}, Section: {section}, "
                f"Relevance: {chunk.relevance_score:.2f})\n{chunk.content}\n"
            )
        return "\n".join(parts)

    def generate_structured_answer(
        self, query: str, retrieved_chunks: List[RetrievedChunk], intent: QueryType
    ) -> dict:
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
        return {
            "answer": answer,
            "format_type": fmt.get(intent, "general"),
            "claims": claims,
            "intent": intent.value,
        }
