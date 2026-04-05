from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from rag_service.api.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse, summary="Run the end-to-end RAG query pipeline")
async def query_rag(request_body: QueryRequest, request: Request) -> QueryResponse:
    query_service = getattr(request.app.state, "query_service", None)
    if query_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query service is not available.",
        )

    return await query_service.answer(request_body.query)
