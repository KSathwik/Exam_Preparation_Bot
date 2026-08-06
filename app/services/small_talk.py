"""Small-talk detection — greetings, thanks, farewells, "how are you".

Routing these through the full RAG pipeline (retrieval, draft LLM call,
reflection LLM call) wastes both latency and money, and the result is
awkward anyway: retrieval finds nothing genuinely relevant to "hi" and the
model ends up either fabricating a description of the uploaded documents
unprompted or asking for clarification in a stilted way. Matched here as an
exact (punctuation/case-insensitive) phrase — deliberately narrow so a real
question is never misfired as small talk.
"""

import re
from typing import Optional

_GREETINGS = {"hi", "hii", "hie", "hey", "hello", "yo", "sup", "hiya"}
_HOW_ARE_YOU = {
    "how are you",
    "how are you doing",
    "how r u",
    "hows it going",
    "how's it going",
    "whats up",
    "what's up",
}
_THANKS = {"thanks", "thank you", "thx", "ty", "thanks a lot", "thank you so much"}
_FAREWELLS = {"bye", "goodbye", "see ya", "see you", "cya"}

_GREETING_REPLY = (
    "Hello! I'm your AI Knowledge Assistant. Upload your documents and ask me anything "
    "about them — definitions, explanations, comparisons, or a detailed walkthrough."
)
_HOW_ARE_YOU_REPLY = (
    "I'm doing well, thanks for asking! Ready whenever you are — what would you like to explore?"
)
_THANKS_REPLY = "You're welcome! Let me know if you have more questions."
_FAREWELL_REPLY = "Goodbye! Have a great day."


def match_small_talk(query: str) -> Optional[str]:
    """Return a canned reply for a small-talk message, or ``None`` if
    ``query`` should go through the normal pipeline."""
    normalized = re.sub(r"[!.?,]+$", "", query.strip().lower()).strip()
    if not normalized:
        return None
    if normalized in _GREETINGS:
        return _GREETING_REPLY
    if normalized in _HOW_ARE_YOU:
        return _HOW_ARE_YOU_REPLY
    if normalized in _THANKS:
        return _THANKS_REPLY
    if normalized in _FAREWELLS:
        return _FAREWELL_REPLY
    return None
