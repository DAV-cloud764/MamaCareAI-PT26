"""Tests for the Gemini translation adapter."""

from __future__ import annotations

import json

import httpx
import pytest
from backend.modules.pipeline.adapters.translation.gemini_translator import (
    GeminiTranslator,
)
from backend.modules.pipeline.config import PipelineSettings
from backend.modules.pipeline.container import build_translator
from backend.modules.pipeline.domain.errors import (
    PermanentError,
    ProviderRateLimited,
    TranslationError,
)


def _response(translations: list[str], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(translations)}]
                    }
                }
            ]
        },
    )


def test_translates_batches_in_order() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        prompt = payload["contents"][0]["parts"][0]["text"]
        inputs = json.loads(prompt.split("Input:\n", 1)[1])
        return _response([f"sw:{text}" for text in inputs])

    translator = GeminiTranslator(
        api_key="secret",
        model="gemini-test",
        batch_size=2,
        transport=httpx.MockTransport(handler),
    )

    result = translator.translate_batch(
        ["one", "two", "three"], source_language="en", target_language="sw"
    )

    assert [item.text for item in result] == ["sw:one", "sw:two", "sw:three"]
    assert len(requests) == 2
    assert requests[0].headers["x-goog-api-key"] == "secret"
    assert requests[0].url.path.endswith("/models/gemini-test:generateContent")
    assert translator.engine_name == "gemini:gemini-test"


def test_rate_limit_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "12"})

    translator = GeminiTranslator(
        api_key="secret", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ProviderRateLimited) as exc_info:
        translator.translate_batch(["one"], source_language="en")
    assert exc_info.value.retry_after_seconds == 12


def test_bad_key_is_permanent_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"error": {"message": "API key not valid. Please pass a valid API key."}},
        )

    translator = GeminiTranslator(
        api_key="secret", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(PermanentError, match="API key not valid"):
        translator.translate_batch(["one"], source_language="en")


def test_rejects_wrong_number_of_translations() -> None:
    translator = GeminiTranslator(
        api_key="secret",
        transport=httpx.MockTransport(lambda request: _response([])),
    )
    with pytest.raises(TranslationError, match="different number"):
        translator.translate_batch(["one"], source_language="en")


def test_supports_only_english_swahili_pair() -> None:
    translator = GeminiTranslator(
        api_key="secret",
        transport=httpx.MockTransport(lambda request: _response([])),
    )
    assert translator.supports("en", "sw")
    assert translator.supports("sw", "en")
    assert not translator.supports("fr", "sw")
    assert not translator.supports("en", "en")


def test_container_builds_gemini_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "current-gemini-key")
    monkeypatch.setenv("PIPELINE_TRANSLATION_API_KEY", "stale-generic-key")

    translator = build_translator(
        PipelineSettings(translation_engine="gemini", gemini_model="gemini-test")
    )

    assert isinstance(translator, GeminiTranslator)
    assert translator.engine_name == "gemini:gemini-test"
    assert translator._client.headers["x-goog-api-key"] == "current-gemini-key"
