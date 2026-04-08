from __future__ import annotations

import re
from dataclasses import dataclass

from fastapi import Request

from rag_service.core.config import Settings
from rag_service.core.exceptions import AuthenticationError, PromptInjectionError


@dataclass(frozen=True)
class SecurityAssessment:
    sanitized_query: str
    warnings: list[str]


_SUSPICIOUS_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"ignore\s+previous\s+instructions",
        r"reveal\s+the\s+system\s+prompt",
        r"developer\s+message",
        r"system\s+prompt",
        r"<system>",
        r"override\s+your\s+instructions",
    ]
]


def sanitize_query(query: str, settings: Settings) -> SecurityAssessment:
    cleaned = " ".join(query.replace("\x00", " ").split())
    cleaned = cleaned[: settings.security.max_query_characters].strip()
    warnings: list[str] = []

    suspicious_matches = [pattern for pattern in _SUSPICIOUS_PATTERNS if pattern.search(cleaned)]
    if not suspicious_matches:
        return SecurityAssessment(sanitized_query=cleaned, warnings=warnings)

    action = settings.security.prompt_injection_action.lower()
    if action == "block":
        raise PromptInjectionError()

    sanitized = cleaned
    for pattern in suspicious_matches:
        sanitized = pattern.sub("", sanitized)
    sanitized = " ".join(sanitized.split()).strip()
    if not sanitized:
        raise PromptInjectionError("The submitted query only contained blocked instruction patterns.")

    warnings.append("Potential instruction-injection content was stripped from the query.")
    return SecurityAssessment(sanitized_query=sanitized, warnings=warnings)


def verify_api_key(request: Request, settings: Settings) -> None:
    if not settings.security.require_api_key:
        return

    header_name = settings.security.auth_header_name
    presented_key = request.headers.get(header_name)
    if not presented_key:
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            presented_key = authorization.split(" ", 1)[1].strip()

    if not presented_key or presented_key not in settings.security.api_keys:
        raise AuthenticationError("A valid API key is required for this endpoint.")
