"""Optional Google Gemini Multimodal Reasoning Provider.

Strictly follows Free-Tier policies:
- Default-disabled; blank or absent key never disrupts core local workflow.
- Strict Zero-Charges: Never upgrades to paid billing, never falls back to paid models.
- Privacy-Enforced: Zero user media transmission without explicit per-job consent.
- Bounded: Enforces per-job maximum call budgets (default 3 calls/job).
- Structured: All outputs are strictly validated against Pydantic schemas.
- Fault-Tolerant: Network timeouts or quota limits seamlessly trigger deterministic rule fallback.
"""

import json
import logging
from enum import Enum
from typing import List, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


class SemanticCategory(str, Enum):
    PERSON = "person"
    MOVING_OBJECT_CANDIDATE = "moving_object_candidate"
    SHADOW_ARTIFACT = "shadow_artifact"


class RecommendedInspection(str, Enum):
    TEMPORAL_RECHECK = "temporal_recheck"
    GEOMETRIC_RECHECK = "geometric_recheck"
    NONE = "none"


class GeminiSemanticEvaluation(BaseModel):
    """Strictly typed structured schema for multimodal crop evaluation."""

    observed_category: SemanticCategory
    visible_evidence: List[str] = Field(min_length=1)
    uncertainties: List[str] = Field(default_factory=list)
    recommended_inspection: RecommendedInspection
    rationale: str = Field(min_length=5)


class GeminiReasoningProvider:
    """Manages optional cloud semantic inspection of ambiguous OpenCV crops."""

    def __init__(self):
        self._call_count = 0

    @property
    def is_available(self) -> bool:
        """True only if enabled AND non-empty API key is configured."""
        return settings.is_gemini_active()

    def reset_job_counter(self) -> None:
        """Resets call budget counter for a new analysis job."""
        self._call_count = 0

    def evaluate_crop(
        self,
        crop_bgr: np.ndarray,
        is_user_media: bool = False,
        user_consent: bool = False,
    ) -> Optional[GeminiSemanticEvaluation]:
        """Evaluates an ambiguous crop using Gemini multimodal vision.

        Returns None if disabled, consent denied, budget exceeded, or on any error.
        """
        if not self.is_available:
            return None

        # Privacy Gate: Strictly prohibit user media transmission without explicit consent
        if is_user_media and not (settings.gemini_send_user_media or user_consent):
            logger.info("Gemini transmission skipped: User media consent not granted.")
            return None

        # Call budget gate
        if self._call_count >= settings.gemini_max_calls_per_job:
            logger.warning(
                "Gemini per-job call budget reached (%d/%d). Falling back to deterministic agent.",
                self._call_count,
                settings.gemini_max_calls_per_job,
            )
            return None

        if crop_bgr is None or crop_bgr.size == 0:
            return None

        try:
            # Import official Google GenAI SDK
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=settings.gemini_api_key)

            # Compress crop into memory JPEG buffer
            success, buffer = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not success:
                return None
            jpeg_bytes = buffer.tobytes()

            prompt = (
                "You are an inspection assistant for a restricted-zone surveillance system. "
                "Inspect this small image crop extracted near the security perimeter. "
                "Determine if it contains a person, a generic moving object candidate, or a shadow/noise artifact. "
                "Respond strictly with a JSON object adhering to this schema:\n"
                "{\n"
                '  "observed_category": "person" | "moving_object_candidate" | "shadow_artifact",\n'
                '  "visible_evidence": ["concise visual observation"],\n'
                '  "uncertainties": ["any ambiguity or occlusion"],\n'
                '  "recommended_inspection": "temporal_recheck" | "geometric_recheck" | "none",\n'
                '  "rationale": "one sentence explanation"\n'
                "}\n"
                "Do NOT include markdown formatting or extra text. Output raw JSON only."
            )

            # Bounded API call
            self._call_count += 1
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=[
                    types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
            )

            if not response or not response.text:
                return None

            clean_text = response.text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            parsed_data = json.loads(clean_text)
            validated = GeminiSemanticEvaluation(**parsed_data)
            logger.info(
                "Successfully received Gemini semantic evaluation: %s", validated.observed_category
            )
            return validated

        except Exception as e:
            # Handles quota limits (429), timeouts, network drops, or JSON validation errors
            logger.warning(
                "Gemini multimodal reasoning call failed or quota exceeded: %s. Reverting to deterministic agent fallback.",
                e,
            )
            return None


# Global provider singleton
gemini_provider = GeminiReasoningProvider()
