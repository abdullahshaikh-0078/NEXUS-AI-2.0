from __future__ import annotations

from fastapi.testclient import TestClient

from rag_service.api.app import create_app
from rag_service.api.schemas import QueryResponse
from rag_service.core.metrics import MetricsRegistry


class FakeQueryService:
    async def answer(self, query: str) -> QueryResponse:
        return QueryResponse(
            answer=f"Answer for {query}",
            citations=[
                {
                    "id": "doc-1",
                    "text": "Grounded evidence",
                    "source": "doc1.txt",
                    "score": 0.91,
                    "title": "Doc One",
                    "section_title": "Overview",
                    "source_path": "data/raw/doc1.txt",
                    "reference_text": "Doc One | section: Overview | source: doc1.txt | score: 0.91",
                }
            ],
            latency_ms=123.4,
            confidence_score=0.88,
            confidence_label="high",
            references_markdown="- [1] Doc One",
            stage_latencies_ms={"retrieval": 45.0, "generation": 30.0},
            debug_documents=[
                {
                    "chunk_id": "doc-1:0001",
                    "text": "Grounded evidence",
                    "source": "doc1.txt",
                    "score": 0.91,
                    "title": "Doc One",
                    "section_title": "Overview",
                    "source_path": "data/raw/doc1.txt",
                }
            ],
            cache_hit=False,
        )


def test_query_endpoint_returns_structured_payload() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.query_service = FakeQueryService()
        response = client.post("/api/v1/query", json={"query": "What is RAG?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Answer for What is RAG?"
    assert payload["citations"][0]["source"] == "doc1.txt"
    assert payload["confidence_label"] == "high"


def test_query_endpoint_rejects_invalid_payload() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/api/v1/query", json={"query": ""})

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid request payload."


def test_metrics_endpoint_returns_runtime_snapshot() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.metrics_registry = MetricsRegistry()
        app.state.metrics_registry.record_stage("retrieval", 42.0)
        app.state.metrics_registry.record_success(
            total_latency_ms=120.0,
            confidence_score=0.81,
            citation_count=2,
            cache_hit=False,
        )
        response = client.get("/api/v1/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_requests"] == 1
    assert payload["stage_metrics"]["retrieval"]["count"] == 1
