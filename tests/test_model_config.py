import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

import model_config


def test_default_configuration(monkeypatch):
    monkeypatch.setattr(model_config, "dotenv_values", lambda *a, **k: {"OPENAI_API_KEY": "dummy-test-key"})
    monkeypatch.setattr(model_config, "ChatOpenAI", lambda **kwargs: kwargs)
    model = model_config.create_model(max_tokens=64)
    assert model["model"] == "gpt-5.6-terra"
    assert model["base_url"] == "https://shuxin79.top/v1"
    assert model["use_responses_api"] is True
    assert model["output_version"] == "responses/v1"
    assert model["store"] is False
    assert model["max_retries"] == 0
    assert model["max_tokens"] == 64


def test_missing_key_does_not_use_global_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-global-key")
    monkeypatch.setattr(model_config, "dotenv_values", lambda *a, **k: {})
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        model_config.create_model()


@pytest.mark.parametrize("url", ["http://shuxin79.top/v1", "https://user:pass@shuxin79.top/v1", "https://shuxin79.top/v1?key=example"])
def test_rejects_unsafe_url(monkeypatch, url):
    monkeypatch.setattr(model_config, "dotenv_values", lambda *a, **k: {"OPENAI_API_KEY": "dummy-test-key", "OPENAI_BASE_URL": url})
    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        model_config.create_model()


@pytest.mark.parametrize("message_type", [AIMessage, AIMessageChunk])
def test_responses_text_blocks(message_type):
    message = message_type(content=[{"type": "text", "text": "OK"}])
    assert message.text == "OK"
