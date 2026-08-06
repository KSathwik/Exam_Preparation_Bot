"""Response format enums and reusable, format-specific prompt templates.

This is the presentation axis of the pipeline — orthogonal to `QueryType`
(the "what is this question about" axis used for retrieval tuning in
`retriever.py`). A `ResponseFormat` answers "how should the answer look" and
drives both the drafting LLM's structure instructions (`FormatTemplate.
prompt_instructions`) and the post-generation `ResponseFormatter` (see
`response_formatter.py`). Adding a new format later (mind maps, cheat sheets,
case studies, ...) means adding one enum member and one `RESPONSE_FORMAT_
TEMPLATES` entry here — nothing else in the pipeline needs to change.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class ResponseFormat(str, Enum):
    """How the answer should be presented — independent of what it's about."""

    DETAILED_EXPLANATION = "detailed_explanation"
    SIMPLE_EXPLANATION = "simple_explanation"
    SUMMARY = "summary"
    KEY_POINTS = "key_points"
    REVISION_NOTES = "revision_notes"
    DEFINITION = "definition"
    COMPARISON = "comparison"
    PROS_CONS = "pros_cons"
    STEPS = "steps"
    TIMELINE = "timeline"
    FLOWCHART = "flowchart"
    FLASHCARDS = "flashcards"
    MCQ = "mcq"
    EXAM_QUESTIONS = "exam_questions"
    INTERVIEW_QUESTIONS = "interview_questions"
    VIVA_QUESTIONS = "viva_questions"
    ONE_LINE = "one_line"
    TWO_MARK = "two_mark"
    FIVE_MARK = "five_mark"
    TEN_MARK = "ten_mark"
    GENERAL = "general"


class ResponseLength(str, Enum):
    """Target answer length, enforced via the LLM call's max_tokens ceiling
    (see llm_interface.py) — a real, enforceable lever, not just a prompt
    suggestion the model can ignore."""

    ULTRA_SHORT = "ultra_short"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    VERY_LONG = "very_long"


# Approximate token ceilings per length tier — deliberately generous rather
# than tight, since truncating an LLM response mid-sentence via max_tokens is
# far worse UX than a slightly-longer-than-ideal answer. The real length
# discipline comes from the prompt instructions below; this is a backstop.
LENGTH_MAX_TOKENS: Dict[ResponseLength, int] = {
    ResponseLength.ULTRA_SHORT: 80,
    ResponseLength.SHORT: 300,
    ResponseLength.MEDIUM: 600,
    ResponseLength.LONG: 1000,
    ResponseLength.VERY_LONG: 1600,
}


@dataclass(frozen=True)
class FormatTemplate:
    """A reusable, format-specific instruction set — the single source of
    truth for "how should this look" fed into both drafting (structure) and
    reflection (structure-conformance check)."""

    prompt_instructions: str
    structure_note: str
    default_length: ResponseLength


RESPONSE_FORMAT_TEMPLATES: Dict[ResponseFormat, FormatTemplate] = {
    ResponseFormat.DETAILED_EXPLANATION: FormatTemplate(
        prompt_instructions=(
            "Write a detailed explanation with clear Markdown headings (##) separating each "
            "facet of the topic. Under each heading, use short paragraphs (2-4 sentences) or "
            "bullet points where the content is a genuinely unordered set of items. Bold key "
            "terms the first time they're introduced. Avoid long uninterrupted paragraphs and "
            "repetitive restatement."
        ),
        structure_note="Headings separating facets, short paragraphs/bullets under each, key terms bolded.",
        default_length=ResponseLength.LONG,
    ),
    ResponseFormat.SIMPLE_EXPLANATION: FormatTemplate(
        prompt_instructions=(
            "Explain this in plain, beginner-friendly language — assume no prior background. "
            "Avoid jargon; when a technical term is unavoidable, define it immediately in "
            "parentheses. Use a short, concrete everyday example or analogy. Keep paragraphs "
            "very short (1-3 sentences)."
        ),
        structure_note="Beginner-friendly language, jargon defined inline, includes a simple example.",
        default_length=ResponseLength.SHORT,
    ),
    ResponseFormat.SUMMARY: FormatTemplate(
        prompt_instructions=(
            "Summarize in exactly 3-6 concise sentences covering only the core ideas. No "
            "headings, no bullet list — a short flowing summary. Do not pad with an "
            "introduction or restate the question."
        ),
        structure_note="3-6 concise sentences, core ideas only, no headings or list.",
        default_length=ResponseLength.SHORT,
    ),
    ResponseFormat.KEY_POINTS: FormatTemplate(
        prompt_instructions=(
            "Respond ONLY as a numbered or bulleted list of 5-8 key points (fewer if the "
            "material genuinely doesn't support that many — never pad to hit the count). Each "
            "point must be 1-2 short sentences — never a paragraph. Each point MUST start on "
            "its own new line with a real Markdown list marker (`1.`/`-`) — never run multiple "
            "points together in one paragraph of continuous text. No introduction or closing "
            "sentence outside the list. Bold the key term or idea at the start of each point."
        ),
        structure_note="Numbered/bulleted list of 5-8 points, each 1-2 short sentences, no prose wrapper.",
        default_length=ResponseLength.SHORT,
    ),
    ResponseFormat.REVISION_NOTES: FormatTemplate(
        prompt_instructions=(
            "Produce quick revision notes: short Markdown headings (##) for each sub-topic, "
            "each followed by 2-5 short bullet points (fragments are fine, full sentences not "
            "required) — optimized for a fast last-minute scan, not for reading prose."
        ),
        structure_note="Headings per sub-topic, short bullet fragments underneath, scannable not prose.",
        default_length=ResponseLength.MEDIUM,
    ),
    ResponseFormat.DEFINITION: FormatTemplate(
        prompt_instructions=(
            "Start with the term in bold, then give a clear one-sentence definition, followed "
            "by one or two supporting sentences with context or a defining characteristic. No "
            "headings or bullet list — a short, tight paragraph."
        ),
        structure_note="Bolded term, one-sentence definition, 1-2 supporting sentences, no list/headings.",
        default_length=ResponseLength.SHORT,
    ),
    ResponseFormat.COMPARISON: FormatTemplate(
        prompt_instructions=(
            "Respond as a Markdown table contrasting the items, with rows for each relevant "
            "aspect and one column per item being compared (e.g. `| Aspect | X | Y |`). Add "
            "at most one sentence before the table for context and one sentence after "
            "highlighting the key takeaway — no other prose."
        ),
        structure_note="Markdown comparison table (aspect rows x item columns), minimal surrounding prose.",
        default_length=ResponseLength.MEDIUM,
    ),
    ResponseFormat.PROS_CONS: FormatTemplate(
        prompt_instructions=(
            "Respond with two clearly separated Markdown sections headed `## Advantages` and "
            "`## Disadvantages`, each a bullet list of concise points (1 sentence each). No "
            "other sections."
        ),
        structure_note="Two headed sections (Advantages/Disadvantages), each a concise bullet list.",
        default_length=ResponseLength.MEDIUM,
    ),
    ResponseFormat.STEPS: FormatTemplate(
        prompt_instructions=(
            "Respond as a numbered list of steps in the order they occur, each on its own new "
            "line with a real Markdown list marker — never run steps together in one paragraph "
            "of continuous text. Each step is one short sentence starting with an action verb "
            "where possible. No prose paragraphs before or between steps."
        ),
        structure_note="Numbered steps in order, one short sentence each, no surrounding prose.",
        default_length=ResponseLength.MEDIUM,
    ),
    ResponseFormat.TIMELINE: FormatTemplate(
        prompt_instructions=(
            "Respond as a chronological timeline: one bullet per date/period/stage, each on its "
            "own new line — never run entries together in one paragraph — in chronological "
            "order, formatted as `**<date/period>:** <what happened, one sentence>`. No "
            "surrounding prose."
        ),
        structure_note="Chronological bullet list, one date/period per bullet, no surrounding prose.",
        default_length=ResponseLength.MEDIUM,
    ),
    ResponseFormat.FLOWCHART: FormatTemplate(
        prompt_instructions=(
            "Respond as a text-based flowchart using arrows to show flow, e.g.:\n"
            "Start -> Step A -> Decision? -> Step B -> End\n"
            "Put each stage on its own line inside a fenced code block, using indentation for "
            "branches. Add at most one sentence of context before the diagram — no other prose."
        ),
        structure_note="Text-based arrow/indentation flowchart in a fenced code block, minimal prose.",
        default_length=ResponseLength.MEDIUM,
    ),
    ResponseFormat.FLASHCARDS: FormatTemplate(
        prompt_instructions=(
            "Produce 5-8 flashcards (fewer if the material doesn't support that many), one "
            "concept per card, each formatted exactly as:\n"
            "**Q:** <question>\n**A:** <answer>\n\n---\n\n"
            "One concept per card — do not combine multiple facts into one card. No "
            "introduction or closing text outside the cards."
        ),
        structure_note="5-8 **Q:**/**A:** pairs, one concept each, separated by `---`, nothing else.",
        default_length=ResponseLength.LONG,
    ),
    ResponseFormat.MCQ: FormatTemplate(
        prompt_instructions=(
            "Generate 5 numbered multiple-choice questions — no more, even if the material "
            "could support additional ones; pick the 5 most exam-relevant. Each question and "
            "each of its options MUST be on its own new line — never run a question, its "
            "options, and its answer together as continuous prose. Each question has exactly 4 "
            "lettered options (A-D), followed on the next line by `**Answer:** <letter> — "
            "<one-sentence explanation of why it's correct>`. No introduction or closing text."
        ),
        structure_note="Exactly 5 numbered MCQs, 4 lettered options each, bolded answer + brief reason.",
        default_length=ResponseLength.LONG,
    ),
    ResponseFormat.EXAM_QUESTIONS: FormatTemplate(
        prompt_instructions=(
            "Generate the 5-8 most likely exam questions based on this material, as a numbered "
            "list — each question, and the hint line beneath it, MUST start on its own new "
            "line; never combine multiple questions into one paragraph of running text, and "
            "never write a full combined answer key at the end instead of per-question hints. "
            "After each question, add one short line (in italics) noting what a strong answer "
            "should cover — not a full answer, just the key points to hit."
        ),
        structure_note="5-8 numbered likely exam questions, each with a short italic hint of key points.",
        default_length=ResponseLength.LONG,
    ),
    ResponseFormat.INTERVIEW_QUESTIONS: FormatTemplate(
        prompt_instructions=(
            "Generate the 5-8 most likely interview questions on this topic, as a numbered "
            "list — each on its own new line, never run together as one paragraph — each "
            "followed by a one-sentence hint of what a strong answer should emphasize."
        ),
        structure_note="5-8 numbered interview questions, each with a one-sentence answer hint.",
        default_length=ResponseLength.LONG,
    ),
    ResponseFormat.VIVA_QUESTIONS: FormatTemplate(
        prompt_instructions=(
            "Generate the 5-8 most likely viva (oral exam) questions on this topic, as a "
            "numbered list — each on its own new line, never run together as one paragraph — "
            "each followed by a one-sentence hint of what a strong verbal answer should cover."
        ),
        structure_note="5-8 numbered viva questions, each with a one-sentence answer hint.",
        default_length=ResponseLength.LONG,
    ),
    ResponseFormat.ONE_LINE: FormatTemplate(
        prompt_instructions=(
            "Respond in EXACTLY one sentence — no more. No preamble, no elaboration, just the "
            "single most important sentence that answers the question."
        ),
        structure_note="Exactly one sentence, nothing else.",
        default_length=ResponseLength.ULTRA_SHORT,
    ),
    ResponseFormat.TWO_MARK: FormatTemplate(
        prompt_instructions=(
            "Answer as a student would for a 2-mark exam question: 2-3 concise sentences "
            "covering only the essential point(s). No headings, no padding."
        ),
        structure_note="2-3 concise sentences, essential points only.",
        default_length=ResponseLength.SHORT,
    ),
    ResponseFormat.FIVE_MARK: FormatTemplate(
        prompt_instructions=(
            "Answer as a student would for a 5-mark exam question: a moderately detailed, "
            "well-structured answer of 2-3 short paragraphs or a short bulleted breakdown, "
            "covering the main points with brief supporting detail — not exhaustive."
        ),
        structure_note="Moderately detailed, 2-3 short paragraphs or a short bulleted breakdown.",
        default_length=ResponseLength.MEDIUM,
    ),
    ResponseFormat.TEN_MARK: FormatTemplate(
        prompt_instructions=(
            "Answer as a student would for a 10-mark exam question: a comprehensive, "
            "well-structured answer with Markdown headings, examples where relevant, and a "
            "brief concluding sentence. This is the one case where thoroughness is expected — "
            "still avoid padding or repetition."
        ),
        structure_note="Comprehensive with headings, examples, and a brief conclusion.",
        default_length=ResponseLength.LONG,
    ),
    ResponseFormat.GENERAL: FormatTemplate(
        prompt_instructions=(
            "Answer clearly and concisely, using short paragraphs or a bullet list — whichever "
            "fits the content better. Avoid padding, repetition, and unnecessary introductions."
        ),
        structure_note="Whichever of short paragraphs or a bullet list fits the content — concise either way.",
        default_length=ResponseLength.MEDIUM,
    ),
}
