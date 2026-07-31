"""Deterministic post-generation formatting — the "dedicated formatting
step" that runs after the LLM has drafted (and reflection has reviewed) an
answer, right before it's returned to the user.

Deliberately narrow in scope: this can safely enforce length and strip
boilerplate without an LLM, but it must NOT attempt to restructure arbitrary
prose into a table/flashcard-deck/etc. after the fact — that would itself
require an LLM call (defeating the point) and risks silently mangling
meaning. Real structural correctness comes from format-aware generation (see
`response_formats.py`'s prompt_instructions, fed into the drafting call) —
this stage is the safety net and consistency layer on top of that, not a
replacement for it.
"""

import re

from .response_formats import ResponseFormat
from .validator import _split_sentences

# Common LLM preambles that add nothing for a student reading an answer —
# stripped regardless of format, since "avoid excessive introductions" is a
# blanket formatting standard, not a per-format concern.
_PREAMBLE_PATTERNS = [
    re.compile(r"^(certainly|sure|of course)!?\s*,?\s*", re.IGNORECASE),
    re.compile(r"^(here'?s|here is)\s+(a|an|the)?\s*", re.IGNORECASE),
    re.compile(r"^based on (the|your) (provided )?document(s)?,?\s*", re.IGNORECASE),
    re.compile(r"^according to (the|your) document(s)?,?\s*", re.IGNORECASE),
]

_TRUNCATE_SENTENCE_LIMITS = {
    ResponseFormat.ONE_LINE: 1,
    ResponseFormat.TWO_MARK: 3,
}


def format_response(text: str, response_format: ResponseFormat) -> str:
    """Normalize a final answer for its response_format. Safe, lossless
    operations only — see module docstring."""
    text = _strip_boilerplate_preamble(text)
    limit = _TRUNCATE_SENTENCE_LIMITS.get(response_format)
    if limit is not None:
        text = _truncate_to_sentences(text, limit)
    return text.strip()


def _strip_boilerplate_preamble(text: str) -> str:
    stripped = text.lstrip()
    for pattern in _PREAMBLE_PATTERNS:
        new = pattern.sub("", stripped, count=1)
        if new != stripped:
            # Re-capitalize the new leading word — stripping "Here's the "
            # from "Here's the mitochondria is..." would otherwise leave a
            # lowercase sentence start.
            stripped = new[:1].upper() + new[1:] if new else new
            break
    return stripped


def _truncate_to_sentences(text: str, max_sentences: int) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= max_sentences:
        return text
    return " ".join(sentences[:max_sentences])
