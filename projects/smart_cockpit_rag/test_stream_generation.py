from types import SimpleNamespace as NS
from unittest.mock import MagicMock
import pytest
from generation import stream_with_client

def chunk(text=None, reason=None):
    return NS(model="offline-model", usage=None, choices=[
        NS(delta=NS(content=text), finish_reason=reason)])

def make_client(chunks):
    client, stream = MagicMock(), MagicMock()
    stream.__enter__.return_value = stream
    stream.__iter__.return_value = iter(chunks)
    client.chat.completions.create.return_value = stream
    return client, stream

@pytest.mark.parametrize("usage_at_finish", [False, True])
def test_stream_collects_text_and_usage(usage_at_finish):
    usage = NS(model_dump=lambda: {"total_tokens": 3})
    end = chunk(reason="stop")
    chunks = [chunk(), chunk("API "), chunk("OK"), end]
    if usage_at_finish:
        end.usage = usage
    else:
        chunks.append(NS(model="offline-model", choices=[], usage=usage))
    client, stream = make_client(chunks)
    events = list(stream_with_client([], client, "test-model", json_mode=True))
    assert [e["type"] for e in events] == ["delta", "delta", "generation_done"]
    assert events[-1] == {"type": "generation_done", "answer": "API OK",
                          "model": "offline-model", "usage": {"total_tokens": 3}}
    options = client.chat.completions.create.call_args.kwargs
    assert options["stream"] is True and options["response_format"]["type"] == "json_object"
    stream.__exit__.assert_called_once()
    client.close.assert_not_called()

@pytest.mark.parametrize("text, reason", [("x", "length"), ("x", None), (" ", "stop")])
def test_incomplete_stream_is_rejected(text, reason):
    client, stream = make_client([chunk(text, reason)])
    with pytest.raises(ValueError):
        list(stream_with_client([], client, "test-model"))
    stream.__exit__.assert_called_once()
    client.close.assert_not_called()

def test_generator_close_exits_stream():
    client, stream = make_client([chunk("API"), chunk("OK"), chunk(reason="stop")])
    reader = stream_with_client([], client, "test-model")
    client.chat.completions.create.assert_not_called()
    assert next(reader) == {"type": "delta", "text": "API"}
    reader.close()
    stream.__exit__.assert_called_once()
    client.close.assert_not_called()