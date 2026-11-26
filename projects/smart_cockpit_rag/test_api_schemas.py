import pytest
from pydantic import ValidationError
from api_schemas import QuestionRequest

def test_question_request_strips_edges():
    request = QuestionRequest.model_validate(
        {"question": "  无钥匙解锁有什么区别？  "}
    )
    assert request.question == "无钥匙解锁有什么区别？"

@pytest.mark.parametrize("payload", [
    {},
    {"question": ""},
    {"question": "   "},
    {"question": 123},
    {"question": "x" * 2001},
    {"question": "test", "top_k": 10},
])
def test_question_request_rejects_invalid_payload(payload):
    with pytest.raises(ValidationError):
        QuestionRequest.model_validate(payload)