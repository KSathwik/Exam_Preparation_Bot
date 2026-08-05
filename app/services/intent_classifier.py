"""Query intent classification module."""

from typing import Dict, Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import redact_query_for_log, settings

from .models import IntentClassificationResult, QueryType


class IntentClassifier:
    """Classify user query intent into predefined categories."""

    INTENT_TEMPLATES = {
        QueryType.DEFINITION: [
            "What is the definition of",
            "Define this term",
            "What does this mean",
            "Explain the term",
        ],
        QueryType.EXPLAIN: [
            "Explain this in detail",
            "Tell me more about",
            "Detail me about",
            "Elaborate on",
            "Describe how it works",
        ],
        QueryType.COMPARE: [
            "Compare and contrast",
            "What is the difference between",
            "How are they different",
            "Similarities and differences",
            "Compare this with",
        ],
        QueryType.PROCESS: [
            "What are the steps",
            "How does it work step by step",
            "What is the process",
            "Describe the procedure",
            "What is the sequence",
        ],
        QueryType.EXAMPLE: [
            "Give me an example",
            "Provide an example",
            "Can you show an example",
            "Real world example",
            "Example of this",
        ],
        QueryType.DIAGRAM: [
            "Explain this diagram",
            "What does this diagram show",
            "Describe this visual",
            "Explain the figure",
            "What is this illustration",
        ],
        QueryType.VAGUE: [
            "What is the answer",
            "Answer this",
            "What about this",
            "Tell me about this",
        ],
    }

    INTENT_KEYWORDS = {
        QueryType.DEFINITION: ["define", "what is", "meaning", "term", "definition"],
        QueryType.EXPLAIN: ["explain", "detail", "elaborate", "describe", "tell me"],
        QueryType.COMPARE: ["compare", "difference", "vs", "versus", "contrast", "similar"],
        QueryType.PROCESS: ["steps", "process", "how does", "procedure", "sequence"],
        QueryType.EXAMPLE: ["example", "instance", "demonstrate", "show", "case study"],
        QueryType.DIAGRAM: ["diagram", "figure", "visual", "illustration", "chart", "graph"],
        QueryType.HOMEWORK: ["question", "answer", "quiz", "test", "exam"],
    }

    INTENT_PROMPTS = {
        QueryType.DEFINITION: (
            "You are a helpful study assistant.\n"
            "The user is asking for a definition.\n"
            "Provide a clear definition based ONLY on the provided document, written as one "
            "or two flowing paragraphs of prose — not a bulleted list.\n"
            "If the definition is not in the document, say so clearly.\n"
            "Format: Start with the term, then explain it in your own connected sentences."
        ),
        QueryType.EXPLAIN: (
            "You are a helpful study assistant.\n"
            "The user is asking for a detailed explanation.\n"
            "Provide a comprehensive explanation based ONLY on the provided document.\n"
            "Include key concepts, mechanisms, and how things work together.\n"
            "Write it as several connected paragraphs of prose, each covering one facet of "
            "the topic — not a bulleted list and not bold-header sections."
        ),
        QueryType.COMPARE: (
            "You are a helpful study assistant.\n"
            "The user is asking to compare two or more concepts.\n"
            "Provide a clear comparison based ONLY on the provided document, written as "
            "prose paragraphs — one paragraph on similarities (if any), one on the key "
            "differences, and one on when to use each, rather than a bulleted list or table."
        ),
        QueryType.PROCESS: (
            "You are a helpful study assistant.\n"
            "The user is asking about a process or steps.\n"
            "Describe the process based ONLY on the provided document as connected, flowing "
            "paragraphs in the order the stages occur — not a numbered list. Use ordering "
            "words in your sentences (first, next, after that, finally) to keep the sequence "
            "clear instead of enumerating steps."
        ),
        QueryType.EXAMPLE: (
            "You are a helpful study assistant.\n"
            "The user is asking for examples.\n"
            "Provide concrete examples based ONLY on the provided document, introducing and "
            "explaining each one in its own sentence or two of prose rather than a bare list.\n"
            "If examples are not in the document, say so clearly.\n"
            "Explain why each example is relevant."
        ),
        QueryType.DIAGRAM: (
            "You are a helpful study assistant.\n"
            "The user is asking about a diagram or figure.\n"
            "Describe what the diagram shows based ONLY on the provided document, as flowing "
            "prose paragraphs rather than a bulleted breakdown.\n"
            "Explain the key elements and what they represent."
        ),
        QueryType.VAGUE: (
            "You are a helpful study assistant.\n"
            "The user's question is unclear.\n"
            "Based on the provided document content, offer the most relevant information as "
            "prose paragraphs, not a bulleted list.\n"
            "If needed, ask for clarification about what specifically they want to know."
        ),
    }

    def __init__(self, model: Optional[SentenceTransformer] = None):
        """Accept a pre-loaded model from DI to avoid duplicate loading."""
        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(settings.intent_embedding_model)
        self.threshold = settings.intent_classification_threshold
        self._compute_template_embeddings()

    def _compute_template_embeddings(self) -> None:
        self.template_embeddings = {}
        for intent, templates in self.INTENT_TEMPLATES.items():
            embeddings = self.model.encode(templates, convert_to_numpy=True)
            self.template_embeddings[intent] = np.mean(embeddings, axis=0)

    def classify(self, query: str) -> IntentClassificationResult:
        logger.debug(f"[INTENT] Classifying: {redact_query_for_log(query)}  threshold={self.threshold}")
        rule_result = self._classify_by_rules(query)
        if rule_result and rule_result["confidence"] > self.threshold:
            logger.info(
                f"[INTENT] Rule match: intent={rule_result['intent'].value}  "
                f"confidence={rule_result['confidence']:.3f}"
            )
            return self._create_result(query, rule_result)
        semantic_result = self._classify_by_semantics(query)
        logger.info(
            f"[INTENT] Semantic match: intent={semantic_result['intent'].value}  "
            f"confidence={semantic_result['confidence']:.3f}"
        )
        logger.debug(
            f"[INTENT] All scores: { {k.value: f'{v:.3f}' for k, v in semantic_result['all_scores'].items()} }"
        )
        return self._create_result(query, semantic_result)

    def _classify_by_rules(self, query: str) -> Optional[Dict]:
        query_lower = query.lower()
        intent_scores = {}
        has_match = False
        for intent, keywords in self.INTENT_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            if matches > 0:
                intent_scores[intent] = 0.8 + 0.15 * (matches / len(keywords))
                has_match = True
            else:
                intent_scores[intent] = 0.0
        if not has_match:
            return None
        best_intent = max(intent_scores, key=intent_scores.get)
        return {
            "intent": best_intent,
            "confidence": min(intent_scores[best_intent], 0.95),
            "all_scores": intent_scores,
            "method": "rules",
        }

    def _classify_by_semantics(self, query: str) -> Dict:
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        intent_scores = {}
        for intent, tmpl_emb in self.template_embeddings.items():
            sim = cosine_similarity(query_embedding.reshape(1, -1), tmpl_emb.reshape(1, -1))[0][0]
            intent_scores[intent] = float(sim)
        best_intent = max(intent_scores, key=intent_scores.get)
        return {
            "intent": best_intent,
            "confidence": intent_scores[best_intent],
            "all_scores": intent_scores,
            "method": "semantic",
        }

    def _create_result(self, query: str, classification: Dict) -> IntentClassificationResult:
        intent = classification["intent"]
        return IntentClassificationResult(
            query=query,
            primary_intent=intent,
            confidence=classification["confidence"],
            alternative_intents={
                k: v for k, v in classification.get("all_scores", {}).items() if k != intent
            },
            reasoning=f"Classified using {classification.get('method', 'unknown')} method",
        )

    def get_classification_prompt(self, intent: QueryType) -> str:
        return self.INTENT_PROMPTS.get(intent, self.INTENT_PROMPTS[QueryType.EXPLAIN])
