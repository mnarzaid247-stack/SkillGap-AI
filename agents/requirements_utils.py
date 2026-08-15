import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class JobRequirements(BaseModel):
    """Structured requirements extracted from a selected job."""

    required_skills: list[str] = Field(
        default_factory=list
    )

    preferred_skills: list[str] = Field(
        default_factory=list
    )

    frameworks: list[str] = Field(
        default_factory=list,
        description=(
            "Named frameworks, libraries, platforms, "
            "cloud services, and technical tools."
        ),
    )

    soft_skills: list[str] = Field(
        default_factory=list
    )

    experience_requirements: list[str] = Field(
        default_factory=list
    )

    education_requirements: list[str] = Field(
        default_factory=list
    )

    responsibilities: list[str] = Field(
        default_factory=list
    )


def model_to_dict(
    model: BaseModel,
) -> dict[str, Any]:
    """Support Pydantic versions 1 and 2."""

    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def validate_requirements_payload(
    payload: dict[str, Any],
) -> JobRequirements:
    """Support Pydantic versions 1 and 2."""

    if hasattr(
        JobRequirements,
        "model_validate",
    ):
        return JobRequirements.model_validate(
            payload
        )

    return JobRequirements.parse_obj(
        payload
    )


def extract_message_text(
    raw_message: Any,
) -> str:
    """
    Extract text content from a raw LangChain message.
    """

    if raw_message is None:
        return ""

    content = getattr(
        raw_message,
        "content",
        "",
    )

    if isinstance(
        content,
        str,
    ):
        return content.strip()

    if isinstance(
        content,
        list,
    ):
        parts = []

        for block in content:
            if isinstance(
                block,
                str,
            ):
                parts.append(
                    block
                )

            elif isinstance(
                block,
                dict,
            ):
                text = (
                    block.get("text")
                    or block.get("content")
                    or ""
                )

                if text:
                    parts.append(
                        str(text)
                    )

        return "\n".join(
            parts
        ).strip()

    return str(
        content
    ).strip()


def extract_json_object(
    text: str,
) -> dict[str, Any] | None:
    """
    Extract a JSON object from model output without
    making another LLM request.
    """

    if not text:
        return None

    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        payload = json.loads(
            cleaned
        )

        if isinstance(
            payload,
            dict,
        ):
            return payload

    except json.JSONDecodeError:
        pass

    start = cleaned.find(
        "{"
    )

    end = cleaned.rfind(
        "}"
    )

    if (
        start == -1
        or end == -1
        or end <= start
    ):
        return None

    try:
        payload = json.loads(
            cleaned[
                start:end + 1
            ]
        )

        if isinstance(
            payload,
            dict,
        ):
            return payload

    except json.JSONDecodeError:
        return None

    return None


def normalize_list_field(
    value: Any,
) -> list[str]:
    """
    Normalize harmless container-shape differences.

    Does not invent requirements.
    """

    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):
        value = value.strip()

        if not value:
            return []

        return [
            value
        ]

    if not isinstance(
        value,
        list,
    ):
        return []

    cleaned = []
    seen = set()

    for item in value:
        text = str(
            item
        ).strip()

        if not text:
            continue

        key = text.casefold()

        if key in seen:
            continue

        seen.add(
            key
        )

        cleaned.append(
            text
        )

    return cleaned


def normalize_requirements_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize structural differences in a model response.

    This does not create or infer any job requirements.
    """

    normalized = dict(
        payload
    )

    fields = [
        "required_skills",
        "preferred_skills",
        "frameworks",
        "soft_skills",
        "experience_requirements",
        "education_requirements",
        "responsibilities",
    ]

    for field_name in fields:
        normalized[
            field_name
        ] = normalize_list_field(
            normalized.get(
                field_name,
                [],
            )
        )

    return normalized


def safe_error_message(
    error: Exception,
) -> str:
    """
    Return useful validation details without exposing
    provider data or sensitive request contents.
    """

    if isinstance(
        error,
        ValidationError,
    ):
        try:
            details = error.errors()

            if details:
                first = details[0]

                location = ".".join(
                    str(part)
                    for part in first.get(
                        "loc",
                        [],
                    )
                )

                message = first.get(
                    "msg",
                    "Invalid structured output.",
                )

                if location:
                    return (
                        "Requirements structured output "
                        f"failed at '{location}': "
                        f"{message}"
                    )

                return (
                    "Requirements structured output "
                    f"failed: {message}"
                )

        except Exception:
            pass

    return (
        "Requirements extraction failed: "
        f"{type(error).__name__}"
    )