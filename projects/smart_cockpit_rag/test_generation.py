import pytest
from openai import OpenAI
from unittest.mock import Mock
import generation

MESSAGES = [{"role": "user", "content": "test"}]

@pytest.fixture
def client():
    with OpenAI(api_key="offline-key", base_url="https://example.invalid/v1") as item:
        item.chat.completions.create = Mock(return_value=Mock(
            choices=[Mock(finish_reason="stop", message=Mock(content=" API_OK "))],
            model="actual-model", usage=None,
        ))
        yield item

@pytest.mark.parametrize("json_mode", [False, True])
def test_borrowed_client_remains_open(client, json_mode):
    for _ in range(2):
        result = generation.generate_with_client(MESSAGES, client, "test-model", json_mode)
    assert result == {"answer": "API_OK", "model": "actual-model", "usage": None}
    assert client.chat.completions.create.call_count == 2
    options = client.chat.completions.create.call_args.kwargs
    assert options.get("response_format") == ({"type": "json_object"} if json_mode else None)
    assert not client.is_closed()

@pytest.mark.parametrize("reason,text", [("length", "text"), ("stop", " ")])
def test_bad_response_keeps_borrowed_client_open(client, reason, text):
    choice = client.chat.completions.create.return_value.choices[0]
    choice.finish_reason, choice.message.content = reason, text
    with pytest.raises(ValueError):
        generation.generate_with_client(MESSAGES, client, "test-model")
    assert not client.is_closed()

@pytest.mark.parametrize("fail", [False, True])
def test_legacy_closes_client(client, monkeypatch, fail):
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(generation, "build_client", lambda: client)
    if fail:
        client.chat.completions.create.side_effect = RuntimeError("offline failure")
        with pytest.raises(RuntimeError):
            generation.generate_answer(MESSAGES)
    else:
        generation.generate_answer(MESSAGES)
    assert client.is_closed()