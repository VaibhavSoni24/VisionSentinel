"""Unit tests for the optional Gemini multimodal provider, privacy gates, and deterministic fallbacks."""

import numpy as np
import pytest

from app.config import settings
from app.providers.gemini import (
    GeminiReasoningProvider,
    GeminiSemanticEvaluation,
    RecommendedInspection,
    SemanticCategory,
)


@pytest.fixture
def dummy_crop():
    crop = np.zeros((60, 40, 3), dtype=np.uint8)
    crop[10:50, 10:30] = (200, 200, 200)
    return crop


def test_gemini_disabled_by_default(dummy_crop):
    provider = GeminiReasoningProvider()
    assert provider.is_available is False
    # When disabled, evaluate_crop must return None immediately without making any HTTP calls
    result = provider.evaluate_crop(dummy_crop)
    assert result is None


def test_privacy_consent_gate(dummy_crop, monkeypatch):
    provider = GeminiReasoningProvider()
    # Mock settings to simulate enabled provider with a dummy key
    monkeypatch.setattr(settings, "gemini_enabled", True)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key-placeholder")
    monkeypatch.setattr(settings, "gemini_send_user_media", False)

    assert provider.is_available is True

    # User media with user_consent=False must be blocked by privacy gate
    res = provider.evaluate_crop(dummy_crop, is_user_media=True, user_consent=False)
    assert res is None  # Blocked!


def test_call_budget_enforcement(dummy_crop, monkeypatch):
    provider = GeminiReasoningProvider()
    monkeypatch.setattr(settings, "gemini_enabled", True)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key-placeholder")
    monkeypatch.setattr(settings, "gemini_max_calls_per_job", 2)

    # Manually set call count to budget limit
    provider._call_count = 2
    res = provider.evaluate_crop(dummy_crop, is_user_media=False)
    assert res is None  # Blocked by budget!


def test_schema_validation():
    # Valid schema payload
    valid_payload = {
        "observed_category": "person",
        "visible_evidence": ["Upright torso visible near boundary line"],
        "uncertainties": ["Lower legs partially occluded"],
        "recommended_inspection": "temporal_recheck",
        "rationale": "High likelihood of human figure entering perimeter.",
    }
    validated = GeminiSemanticEvaluation(**valid_payload)
    assert validated.observed_category == SemanticCategory.PERSON
    assert validated.recommended_inspection == RecommendedInspection.TEMPORAL_RECHECK

    # Invalid category should fail Pydantic validation
    invalid_payload = valid_payload.copy()
    invalid_payload["observed_category"] = "unsupported_animal"
    with pytest.raises(Exception):
        GeminiSemanticEvaluation(**invalid_payload)
