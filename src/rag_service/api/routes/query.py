from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from rag_service.api.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse, summary="Run the end-to-end RAG query pipeline")
async def query_rag(request_body: QueryRequest, request: Request) -> QueryResponse:
    query_service = getattr(request.app.state, "query_service", None)
    if query_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query service is not available.",
        )

    return await query_service.answer(request_body.query)


@router.post("/query/stream", summary="Stream the end-to-end RAG query pipeline")
async def stream_query_rag(request_body: QueryRequest, request: Request) -> StreamingResponse:
    settings = getattr(request.app.state, "settings", None)
    if settings is not None:
        verify_api_key(request, settings)

    query_service = getattr(request.app.state, "query_service", None)
    if query_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query service is not available.",
        )

    async def event_stream():
        async for event in query_service.stream_answer(request_body.query):
            yield json.dumps(event) + "\n"

    media_type = settings.latency.stream_media_type if settings is not None else "application/x-ndjson"
    return StreamingResponse(event_stream(), media_type=media_type)
