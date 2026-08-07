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
    _OllamaLLM,
    _OpenAILLM,
)
from app.services.models import ChatMessage, ChunkMetadata, QueryType, RetrievedChunk
from app.services.response_formats import ResponseFormat


@pytest.fixture(autouse=True)
def _restore_provider(monkeypatch):
    """Every test overrides llm_provider — restore it afterwards."""
    original = settings.llm_provider
    yield
    monkeypatch.setattr(settings, "llm_provider", original, raising=False)


def _chunk(content="Mitochondria are the powerhouse of the cell.", page=4):
    meta = ChunkMetadata(page_number=page, chunk_index=0, total_chunks=1, file_name="bio.pdf")
    return RetrievedChunk(content=content, metadata=meta, relevance_score=0.9, rank=1)


def test_factory_returns_ollama(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_model", "llama3.2")
    llm = ClaudeInterface()
    assert isinstance(llm, _OllamaLLM)
    assert llm.model == "llama3.2"


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
    llm._client.messages.create = MagicMock(return_value=fake_response)  # type: ignore[attr-defined]

    result = llm._call("system prompt", "user message")
    assert result == "42"
    llm._client.messages.create.assert_called_once()  # type: ignore[attr-defined]


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
    llm._client.messages.create = MagicMock(return_value=fake_response)  # type: ignore[attr-defined]

    result = llm._call("system prompt", "user message")
    assert result == "the real answer"


def test_openai_call_uses_client(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    llm = ClaudeInterface()

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="answer text"))]
    fake_response.usage = MagicMock(prompt_tokens=5, completion_tokens=2)
    llm._client.chat.completions.create = MagicMock(return_value=fake_response)  # type: ignore[attr-defined]

    result = llm._call("system prompt", "user message")
    assert result == "answer text"


def test_gemini_call_uses_client(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    llm = ClaudeInterface()

    fake_response = MagicMock()
    fake_response.text = "gemini answer"
    llm._client.models.generate_content = MagicMock(return_value=fake_response)  # type: ignore[attr-defined]

    result = llm._call("system prompt", "user message")
    assert result == "gemini answer"
    llm._client.models.generate_content.assert_called_once()  # type: ignore[attr-defined]


def test_generate_answer_with_claims_parses_json_object(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(  # type: ignore[method-assign]
        return_value=(
            "Here you go: "
            '{"answer": "Mitochondria produce ATP.", '
            '"claims": [{"claim": "Mitochondria produce ATP.", "chunks": [1]}]} trailing'
        )
    )

    result = llm.generate_answer_with_claims(
        "What is a mitochondria?", [_chunk(), _chunk(page=7)], QueryType.DEFINITION
    )

    assert result == {
        "answer": "Mitochondria produce ATP.",
        "claims": [{"claim": "Mitochondria produce ATP.", "chunks": [1]}],
    }


def test_generate_answer_with_claims_tolerates_plain_string_claims(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(return_value='{"answer": "The answer.", "claims": ["Claim one", "Claim two"]}')  # type: ignore[method-assign]

    result = llm.generate_answer_with_claims("q", [_chunk()], QueryType.DEFINITION)

    assert result == {
        "answer": "The answer.",
        "claims": [{"claim": "Claim one", "chunks": []}, {"claim": "Claim two", "chunks": []}],
    }


def test_generate_answer_with_claims_repairs_unescaped_newlines_in_answer(monkeypatch):
    """A heading/list-heavy Markdown answer (e.g. KEY_POINTS, DETAILED_EXPLANATION)
    commonly contains literal newlines between paragraphs/list items that the
    model doesn't reliably escape as \\n despite being told to — a raw control
    character inside a JSON string is invalid and previously broke json.loads
    for the *whole* object, leaking the raw JSON envelope as the visible
    answer (found via live smoke testing of the response-formatting pass)."""
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    raw_with_literal_newlines = (
        '{"answer": "## Heading One\n- Point one\n- Point two\n\n## Heading Two\nMore text.", '
        '"claims": [{"claim": "Point one", "chunks": [1]}]}'
    )
    llm._call = MagicMock(return_value=raw_with_literal_newlines)  # type: ignore[method-assign]

    result = llm.generate_answer_with_claims("q", [_chunk()], QueryType.DEFINITION)

    assert result["answer"] == "## Heading One\n- Point one\n- Point two\n\n## Heading Two\nMore text."
    assert result["claims"] == [{"claim": "Point one", "chunks": [1]}]


def test_generate_answer_with_claims_best_effort_recovers_embedded_quotes_and_js_escapes(monkeypatch):
    """Found via live smoke testing of the MCQ format: a quoted term embedded
    *inside* the answer text (e.g. an "ocean world" reference) breaks the
    whole-object parse, and — because the model also escaped an apostrophe
    JS/Python-style as \\' (not a valid JSON escape) — even the existing
    quote-matching regex recovery's own json.loads rejects the captured
    fragment. Only the structural (", "claims" boundary, not quote-matching)
    best-effort extraction survives both failures, and must produce clean
    text with no JSON envelope/escape sequences leaking into the answer."""
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    raw = (
        '{"answer": "Earth\\\'s Moon formed from Theia. The document calls Earth an '
        '"ocean world".", "claims": [{"claim": "test claim", "chunks": [1]}]}'
    )
    llm._call = MagicMock(return_value=raw)  # type: ignore[method-assign]

    result = llm.generate_answer_with_claims("q", [_chunk()], QueryType.DEFINITION)

    assert result["answer"] == 'Earth\'s Moon formed from Theia. The document calls Earth an "ocean world".'
    assert result["claims"] == []  # claims are sacrificed on this path — only the answer is recovered


def test_generate_answer_with_claims_recovers_partial_answer_from_truncated_response(monkeypatch):
    """Found via live smoke testing of the MCQ format: a response that hits
    max_tokens mid-generation never emits a closing '}' at all, so
    raw.rfind("}") returns -1 — previously that skipped every recovery tier
    entirely (e > s was never true) and fell straight to showing the raw,
    unterminated JSON envelope. A readable partial answer sitting right after
    the opening brace must still be recovered instead."""
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    # No closing brace anywhere — simulates a hard mid-generation cutoff.
    truncated = (
        '{"answer":"1. What percentage of Earth\'s crust is covered by ocean?\\n'
        "A. 50.2%\\nB. 70.8%\\nC. 29.2%\\nD. 90.5%\\n**Answer:** B \\u2014 the excerpt "
        "states 70.8%.\\n\\n2. What causes tectonic activity?\\nA. Solar wind\\nB. Con"
    )
    llm._call = MagicMock(return_value=truncated)  # type: ignore[method-assign]

    result = llm.generate_answer_with_claims("q", [_chunk()], QueryType.DEFINITION)

    assert result["answer"].startswith("1. What percentage of Earth's crust is covered by ocean?")
    assert "2. What causes tectonic activity?" in result["answer"]
    assert not result["answer"].startswith("{")  # never leak the raw JSON envelope
    assert result["claims"] == []


def test_generate_answer_with_claims_recovers_answer_when_claims_malformed(monkeypatch):
    """A malformed claims array (e.g. an unescaped quote mark the model
    didn't escape) breaks whole-object json.loads — the answer field must
    still be recovered rather than losing the turn entirely (regression
    found via live smoke testing of the equivalent extract_claims failure)."""
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    malformed = (
        '{"answer": "Earth has land and ocean.", '
        '"claims": [{"claim": "Earth is an "ocean world"", "chunks": [1]}]}'
    )
    llm._call = MagicMock(return_value=malformed)  # type: ignore[method-assign]

    result = llm.generate_answer_with_claims("q", [_chunk()], QueryType.DEFINITION)

    assert result == {"answer": "Earth has land and ocean.", "claims": []}


def test_generate_answer_with_claims_falls_back_to_raw_text(monkeypatch):
    """When the response isn't JSON at all, the raw text is still used as
    the answer rather than losing the turn entirely."""
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(return_value="Just a plain prose answer, no JSON at all.")  # type: ignore[method-assign]

    result = llm.generate_answer_with_claims("q", [_chunk()], QueryType.DEFINITION)

    assert result == {"answer": "Just a plain prose answer, no JSON at all.", "claims": []}


def test_generate_structured_answer_maps_format_type(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm.generate_answer_with_claims = MagicMock(  # type: ignore[method-assign]
        return_value={
            "answer": "The mitochondria is the powerhouse of the cell.",
            "claims": [{"claim": "The mitochondria is the powerhouse of the cell.", "chunks": [1]}],
        }
    )

    result = llm.generate_structured_answer(
        "What is a mitochondria?", [_chunk()], QueryType.DEFINITION, response_format=ResponseFormat.DEFINITION
    )
    assert result["format_type"] == "definition"
    assert result["claims"] == [{"claim": "The mitochondria is the powerhouse of the cell.", "chunks": [1]}]
    assert result["intent"] == "definition"


def test_reflect_on_answer_parses_json(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(  # type: ignore[method-assign]
        return_value=(
            "Here is my review: "
            '{"revised_answer": "Better answer.", "materially_changed": true, '
            '"should_block": false, "issues_found": ["minor wording"], '
            '"claims": [{"claim": "Better answer.", "chunks": [1]}]}'
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
        "claims": [{"claim": "Better answer.", "chunks": [1]}],
    }


def test_reflect_on_answer_falls_back_on_malformed_response(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(return_value="not json at all")  # type: ignore[method-assign]

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
        "claims": None,
    }


def test_reflect_on_answer_falls_back_on_exception(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(side_effect=RuntimeError("provider down"))  # type: ignore[method-assign]

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
        "claims": None,
    }


def test_summarize_conversation_returns_text(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    llm = ClaudeInterface()
    llm._call = MagicMock(return_value="  Discussed mitochondria structure and function.  ")  # type: ignore[method-assign]

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
    llm._call = MagicMock(side_effect=RuntimeError("provider down"))  # type: ignore[method-assign]

    turns = [ChatMessage(role="user", content="hi", timestamp="2024-01-01T00:00:00")]
    assert llm.summarize_conversation(turns) == ""


def test_ollama_call_uses_client(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_model", "llama3.2")
    llm = ClaudeInterface()
    assert isinstance(llm, _OllamaLLM)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"message": {"content": "Ollama answer text"}, "eval_count": 15}
    fake_resp.raise_for_status = MagicMock()
    llm._client.post = MagicMock(return_value=fake_resp)  # type: ignore[method-assign]

    res = llm._call("sys prompt", "user message")
    assert res == "Ollama answer text"
    llm._client.post.assert_called_once()


def test_ollama_call_handles_404_model_not_found(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_model", "invalid-model")
    llm = ClaudeInterface()

    fake_resp = MagicMock()
    fake_resp.status_code = 404
    llm._client.post = MagicMock(return_value=fake_resp)  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="not found at"):
        llm._call("sys", "user")


def test_ollama_check_health_healthy(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_model", "llama3.2")
    llm = ClaudeInterface()

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"models": [{"name": "llama3.2:latest"}, {"name": "mistral"}]}
    llm._client.get = MagicMock(return_value=fake_resp)  # type: ignore[method-assign]

    health = llm.check_health()
    assert health["status"] == "healthy"
    assert health["server_connected"] is True
    assert health["model_available"] is True


def test_ollama_check_health_unhealthy_connection_error(monkeypatch):
    import httpx

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    llm = ClaudeInterface()
    llm._client.get = MagicMock(side_effect=httpx.ConnectError("Connection refused"))  # type: ignore[method-assign]

    health = llm.check_health()
    assert health["status"] == "unhealthy"
    assert health["server_connected"] is False
    assert "Could not connect to Ollama server" in health["error"]


def test_ollama_streaming(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    llm = ClaudeInterface()
    assert isinstance(llm, _OllamaLLM)

    fake_context = MagicMock()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.raise_for_status = MagicMock()
    fake_resp.iter_lines.return_value = [
        '{"message": {"content": "Hello "}}',
        '{"message": {"content": "world!"}}',
    ]
    fake_context.__enter__.return_value = fake_resp
    llm._client.stream = MagicMock(return_value=fake_context)  # type: ignore[method-assign]

    chunks = list(llm._stream_call("sys", "user"))
    assert chunks == ["Hello ", "world!"]
