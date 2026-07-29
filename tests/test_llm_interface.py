"""Tests for the multi-provider LLM interface.

Each provider client is only ever *constructed* here — the underlying SDK's
network call (``_call``) is stubbed, so nothing in this file makes a real
request to Anthropic, OpenAI, or Gemini.
"""

from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.services.llm_interface import (
    ClaudeInterface,
    _AnthropicLLM,
    _GeminiLLM,
    _OpenAILLM,
)
from app.services.models import ChatMessage, ChunkMetadata, QueryType, RetrievedChunk


@pytest.fixture(autouse=True)
def _restore_provider(monkeypatch):
    """Every test overrides llm_provider — restore it afterwards."""
    original = settings.llm_provider
    yield
    monkeypatch.setattr(settings, "llm_provider", original, raising=False)


def _chunk(content="Mitochondria are the powerhouse of the cell.", page=4):
    meta = ChunkMetadata(page_number=page, chunk_index=0, total_chunks=1, file_name="bio.pdf")
    return RetrievedChunk(content=content, metadata=meta, relevance_score=0.9, rank=1)


def test_factory_returns_anthropic(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    assert isinstance(llm, _AnthropicLLM)
    assert llm.model == "claude-sonnet-5"


def test_factory_returns_openai(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    llm = ClaudeInterface()
    assert isinstance(llm, _OpenAILLM)
    assert llm.model == "gpt-4o-mini"


def test_factory_returns_gemini(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    llm = ClaudeInterface()
    assert isinstance(llm, _GeminiLLM)
    assert llm.model == "gemini-2.0-flash"


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "not-a-real-provider")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        ClaudeInterface()


def test_factory_respects_model_name_override(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "model_name", "claude-custom-model")
    llm = ClaudeInterface()
    assert llm.model == "claude-custom-model"


def test_anthropic_call_uses_client(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()

    fake_response = MagicMock()
    fake_response.content = [MagicMock(type="text", text="42")]
    fake_response.usage.input_tokens = 10
    fake_response.usage.output_tokens = 3
    llm._client.messages.create = MagicMock(return_value=fake_response)

    result = llm._call("system prompt", "user message")
    assert result == "42"
    llm._client.messages.create.assert_called_once()


def test_anthropic_call_skips_thinking_blocks(monkeypatch):
    """Current-generation Claude models can emit a ThinkingBlock before the
    text block — _call must pick out the text block rather than blindly
    indexing content[0] (regression: this silently broke extract_claims/
    reflect_on_answer in production when thinking blocks were present)."""
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()

    fake_response = MagicMock()
    thinking_block = MagicMock(type="thinking")
    del thinking_block.text  # a real ThinkingBlock has no .text attribute
    fake_response.content = [thinking_block, MagicMock(type="text", text="the real answer")]
    fake_response.usage.input_tokens = 10
    fake_response.usage.output_tokens = 3
    llm._client.messages.create = MagicMock(return_value=fake_response)

    result = llm._call("system prompt", "user message")
    assert result == "the real answer"


def test_openai_call_uses_client(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    llm = ClaudeInterface()

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="answer text"))]
    fake_response.usage = MagicMock(prompt_tokens=5, completion_tokens=2)
    llm._client.chat.completions.create = MagicMock(return_value=fake_response)

    result = llm._call("system prompt", "user message")
    assert result == "answer text"


def test_gemini_call_uses_client(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    llm = ClaudeInterface()

    fake_response = MagicMock()
    fake_response.text = "gemini answer"
    llm._client.models.generate_content = MagicMock(return_value=fake_response)

    result = llm._call("system prompt", "user message")
    assert result == "gemini answer"
    llm._client.models.generate_content.assert_called_once()


def test_extract_claims_parses_chunk_indexed_json_array(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(
        return_value=(
            "Some text before "
            '[{"claim": "Claim one", "chunks": [1]}, {"claim": "Claim two", "chunks": [1, 2]}] trailing'
        )
    )

    claims = llm.extract_claims("An answer with two claims.", [_chunk(), _chunk(page=7)])
    assert claims == [
        {"claim": "Claim one", "chunks": [1]},
        {"claim": "Claim two", "chunks": [1, 2]},
    ]


def test_extract_claims_recovers_valid_objects_when_array_parse_fails(monkeypatch):
    """A single malformed claim (e.g. the model didn't escape an embedded
    quote mark) breaks whole-array json.loads — the other, individually
    well-formed claim objects should still be recovered rather than
    discarding the entire batch (regression found via live smoke testing)."""
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    malformed = (
        '[{"claim": "Earth is an "ocean world"", "chunks": [1]}, '
        '{"claim": "Land covers 29.2% of the surface", "chunks": [1]}]'
    )
    llm._call = MagicMock(return_value=malformed)

    claims = llm.extract_claims("some answer", [_chunk()])

    assert claims == [{"claim": "Land covers 29.2% of the surface", "chunks": [1]}]


def test_extract_claims_tolerates_plain_string_array(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(return_value='["Claim one", "Claim two"]')

    claims = llm.extract_claims("An answer with two claims.", [_chunk()])
    assert claims == [
        {"claim": "Claim one", "chunks": []},
        {"claim": "Claim two", "chunks": []},
    ]


def test_extract_claims_returns_empty_on_malformed_response(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(return_value="not json at all")

    assert llm.extract_claims("anything", [_chunk()]) == []


def test_generate_structured_answer_maps_format_type(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm.generate_answer = MagicMock(return_value="The mitochondria is the powerhouse of the cell.")
    llm.extract_claims = MagicMock(return_value=["The mitochondria is the powerhouse of the cell."])

    result = llm.generate_structured_answer("What is a mitochondria?", [_chunk()], QueryType.DEFINITION)
    assert result["format_type"] == "definition"
    assert result["claims"] == ["The mitochondria is the powerhouse of the cell."]
    assert result["intent"] == "definition"


def test_reflect_on_answer_parses_json(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(
        return_value=(
            "Here is my review: "
            '{"revised_answer": "Better answer.", "materially_changed": true, '
            '"should_block": false, "issues_found": ["minor wording"]}'
        )
    )

    result = llm.reflect_on_answer(
        query="What is a mitochondria?",
        retrieved_chunks=[_chunk()],
        draft_answer="Original answer.",
        validator_summary="1/1 claims cited, confidence=0.8",
        intent=QueryType.DEFINITION,
    )
    assert result == {
        "revised_answer": "Better answer.",
        "materially_changed": True,
        "should_block": False,
        "issues_found": ["minor wording"],
    }


def test_reflect_on_answer_falls_back_on_malformed_response(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(return_value="not json at all")

    result = llm.reflect_on_answer(
        query="What is a mitochondria?",
        retrieved_chunks=[_chunk()],
        draft_answer="Original answer.",
        validator_summary="",
        intent=QueryType.DEFINITION,
    )
    assert result == {
        "revised_answer": "Original answer.",
        "materially_changed": False,
        "should_block": False,
        "issues_found": [],
    }


def test_reflect_on_answer_falls_back_on_exception(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(side_effect=RuntimeError("provider down"))

    result = llm.reflect_on_answer(
        query="What is a mitochondria?",
        retrieved_chunks=[_chunk()],
        draft_answer="Original answer.",
        validator_summary="",
        intent=QueryType.VAGUE,
    )
    assert result == {
        "revised_answer": "Original answer.",
        "materially_changed": False,
        "should_block": False,
        "issues_found": [],
    }


def test_summarize_conversation_returns_text(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(return_value="  Discussed mitochondria structure and function.  ")

    turns = [
        ChatMessage(role="user", content="What is a mitochondria?", timestamp="2024-01-01T00:00:00"),
        ChatMessage(
            role="assistant", content="It's the powerhouse of the cell.", timestamp="2024-01-01T00:00:01"
        ),
    ]
    summary = llm.summarize_conversation(turns)
    assert summary == "Discussed mitochondria structure and function."


def test_summarize_conversation_returns_empty_on_failure(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(side_effect=RuntimeError("provider down"))

    turns = [ChatMessage(role="user", content="hi", timestamp="2024-01-01T00:00:00")]
    assert llm.summarize_conversation(turns) == ""
