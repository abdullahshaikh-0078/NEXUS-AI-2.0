from __future__ import annotations


class RAGServiceError(Exception):
    status_code = 500
    detail = "Internal service error."

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or self.detail)
        self.detail = detail or self.detail


class AuthenticationError(RAGServiceError):
    status_code = 401
    detail = "Authentication failed."


class PromptInjectionError(RAGServiceError):
    status_code = 400
    detail = "Potential prompt injection attempt detected."


class BackpressureError(RAGServiceError):
    status_code = 503
    detail = "Service is at capacity. Please retry shortly."


class CircuitBreakerOpenError(RAGServiceError):
    status_code = 503
    detail = "A downstream dependency is temporarily unavailable."
