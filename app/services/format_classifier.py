"""Deterministic response-format classification.

Answers "how should this answer look" — orthogonal to `IntentClassifier`
(which answers "what is this question about", for retrieval tuning). No LLM
fallback at all: unlike intent, a format that fails to match any explicit
keyword directive always has a perfectly reasonable answer — the requesting
intent's own default format (see `INTENT_DEFAULT_FORMAT`) — so there is no
query for which paying for an LLM call would improve on that default.
"""

from typing import Dict, List, Optional

from loguru import logger

from .models import QueryType
from .response_formats import ResponseFormat

# Longest-keyword-first within each format's list isn't required — matching
# just needs *a* hit; ties across formats are broken by keyword length (a
# longer, more specific phrase winning over a short, generic one), not by
# insertion order, so e.g. "explain simply" (SIMPLE_EXPLANATION) beats a
# stray "explain" substring that no format list even contains.
FORMAT_KEYWORDS: Dict[ResponseFormat, List[str]] = {
    ResponseFormat.ONE_LINE: ["one line", "one-line", "one sentence", "single sentence"],
    ResponseFormat.TWO_MARK: ["two mark", "2 mark", "2-mark", "two-mark"],
    ResponseFormat.FIVE_MARK: ["five mark", "5 mark", "5-mark", "five-mark"],
    ResponseFormat.TEN_MARK: ["ten mark", "10 mark", "10-mark", "ten-mark"],
    ResponseFormat.FLASHCARDS: ["flashcard", "flash card"],
    ResponseFormat.MCQ: [
        "mcq",
        "multiple choice",
        "multiple-choice",
        "quiz me",
        "generate a quiz",
        "generate quiz",
    ],
    ResponseFormat.VIVA_QUESTIONS: ["viva question", "viva voce"],
    ResponseFormat.INTERVIEW_QUESTIONS: ["interview question"],
    ResponseFormat.EXAM_QUESTIONS: [
        "exam question",
        "important question",
        "likely question",
        "possible question",
    ],
    ResponseFormat.REVISION_NOTES: [
        "revision note",
        "revision notes",
        "quick note",
        "last minute revision",
        "last-minute revision",
    ],
    ResponseFormat.SIMPLE_EXPLANATION: [
        "explain simply",
        "explain like i'm",
        "explain like i am",
        "eli5",
        "easy explanation",
        "in simple terms",
        "beginner explanation",
        "explain it simply",
    ],
    ResponseFormat.PROS_CONS: [
        "advantages and disadvantages",
        "pros and cons",
        "pros & cons",
        "merits and demerits",
    ],
    ResponseFormat.COMPARISON: [
        "difference between",
        "compare and contrast",
        " vs ",
        " vs. ",
        "versus",
        "compare",
    ],
    ResponseFormat.TIMELINE: ["timeline", "history of", "evolution of", "chronology", "chronological"],
    ResponseFormat.FLOWCHART: ["flowchart", "flow chart"],
    ResponseFormat.STEPS: [
        "step by step",
        "step-by-step",
        "the steps",
        "lifecycle",
        "working of",
        "the process of",
    ],
    ResponseFormat.KEY_POINTS: [
        "key points",
        "in points",
        "give me points",
        "list the main points",
        "important points",
        "bullet points",
        "in bullets",
    ],
    ResponseFormat.SUMMARY: ["summarize", "summarise", "short summary", "tl;dr", "tldr", "in summary"],
    ResponseFormat.DEFINITION: ["define ", "definition of", "meaning of", "what is the meaning"],
}


class FormatClassifier:
    """Rule-only response-format classifier — see module docstring for why
    there's deliberately no semantic/LLM fallback tier."""

    # Fallback when no explicit format directive is found in the query.
    INTENT_DEFAULT_FORMAT: Dict[QueryType, ResponseFormat] = {
        QueryType.DEFINITION: ResponseFormat.DEFINITION,
        QueryType.EXPLAIN: ResponseFormat.DETAILED_EXPLANATION,
        QueryType.COMPARE: ResponseFormat.COMPARISON,
        QueryType.PROCESS: ResponseFormat.STEPS,
        QueryType.EXAMPLE: ResponseFormat.DETAILED_EXPLANATION,
        QueryType.DIAGRAM: ResponseFormat.FLOWCHART,
        QueryType.VAGUE: ResponseFormat.GENERAL,
        QueryType.HOMEWORK: ResponseFormat.EXAM_QUESTIONS,
    }

    def classify(self, query: str, intent: QueryType) -> ResponseFormat:
        matched = self._match_keywords(query)
        if matched is not None:
            logger.info(f"[FORMAT] Keyword match: format={matched.value}")
            return matched
        default = self.INTENT_DEFAULT_FORMAT.get(intent, ResponseFormat.GENERAL)
        logger.info(
            f"[FORMAT] No directive found — defaulting from intent={intent.value}: format={default.value}"
        )
        return default

    @staticmethod
    def _match_keywords(query: str) -> Optional[ResponseFormat]:
        query_lower = f" {query.lower()} "  # padded so " vs " etc. can match at string edges
        best_format: Optional[ResponseFormat] = None
        best_len = 0
        for fmt, keywords in FORMAT_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower and len(kw) > best_len:
                    best_format, best_len = fmt, len(kw)
        return best_format
